"""Route at the router layer, write at an EARLIER layer at the subject tokens.

The linear router reads the raw layer-19 state at the prompt boundary. Facts, however,
are recalled earlier, at the subject's tokens (mid-layer MLPs; ROME / Geva et al.), and
every later position can re-read them from there. EarlyWriteCausalLM therefore runs

  pass 1  routing: the router bank with ZERO rows (bit-exact base) decides the fact at
          the boundary. The decision depends only on the prompt prefix and never on the
          rows, so it is cached per prompt (exact).
  pass 2  editing: if fact f was routed, rows[f] is added to the residual stream after
          block `write_layer` at the subject positions of fact f in the prompt
          (`subject_last` = last subject token, `subject_span` = all subject tokens).
          If the subject is not found, the boundary token receives the row (counted).

Unrouted prompts get no write, so they stay exactly base (pass 2 adds zeros through the
router bank and nothing at the write layer).

Artifacts: a normal linear-router artifact plus
    artifact["early_write"] = {"layer": L, "mode": "subject_span" | "subject_last"}
with artifact["rows"] holding the early-write rows. linear_router.load_router_artifact
dispatches here, so every evaluator (official MCF eval, routed generation, held-out
generation, lens eval) loads it unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

EARLY_WRITE_MODES = ("subject_last", "subject_span")


def _block_output(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_output(out, hidden):
    return (hidden, *out[1:]) if isinstance(out, tuple) else hidden


class EarlyWriteCausalLM(nn.Module):
    def __init__(self, base_model, artifact, write_layer, write_mode="subject_span", rows=None):
        super().__init__()
        from linear_router import load_linear_classifier_artifact

        if write_mode not in EARLY_WRITE_MODES:
            raise ValueError(f"write_mode must be one of {EARLY_WRITE_MODES}")
        router_layer = int(artifact["layer"])
        if not 0 <= int(write_layer) <= router_layer:
            raise ValueError(f"write_layer must be in [0, {router_layer}] (router layer)")
        router_artifact = dict(artifact)
        router_artifact.pop("early_write", None)
        router_artifact["rows"] = torch.zeros_like(artifact["rows"])
        router_artifact["write_mode"] = "last"
        router_artifact["edit_mode"] = "add"
        self.router_lm, self.bank = load_linear_classifier_artifact(base_model, router_artifact)
        self.base_model = base_model
        device = next(base_model.parameters()).device
        initial = artifact["rows"] if rows is None else rows
        self.rows = nn.Parameter(initial.detach().float().clone().to(device))
        self.write_layer = int(write_layer)
        self.write_mode = write_mode
        self.subject_patterns = artifact["subject_patterns"]
        self.facts = artifact["facts"]
        self.source_artifact = artifact
        self.route_cache = {}
        self.subject_fallbacks = 0
        self.last_active_fact_indices = []
        self._next_prefix_lengths = None
        self._plan = None
        self._handle = base_model.model.layers[self.write_layer].register_forward_hook(self._hook)

    # ------------------------------------------------------------------ HF-ish surface
    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_association_prefix_lengths(self, lengths):
        self._next_prefix_lengths = torch.as_tensor(lengths, dtype=torch.long)

    def close(self):
        self._handle.remove()
        self.bank.close()

    def counters(self):
        return {"early_write": {"layer": self.write_layer, "mode": self.write_mode},
                "subject_fallbacks": self.subject_fallbacks,
                "cached_routes": len(self.route_cache),
                "cached_routes_active": sum(f >= 0 for f in self.route_cache.values()),
                "router_bank": self.bank.counters()}

    # ---------------------------------------------------------------------- routing
    @staticmethod
    def _boundaries(input_ids, attention_mask, prefix_lengths):
        if prefix_lengths is not None:
            return [int(x) for x in prefix_lengths.tolist()]
        if attention_mask is not None:
            mask = attention_mask.bool()
            idx = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
            return [int(x) + 1 for x in idx.masked_fill(~mask, -1).max(1).values.tolist()]
        return [input_ids.shape[1]] * input_ids.shape[0]

    def _prompt(self, ids_row, mask_row, boundary):
        return [(p, int(t)) for p, t in enumerate(ids_row[:boundary])
                if mask_row is None or bool(mask_row[p])]

    def _route(self, input_ids, attention_mask, boundaries):
        ids = input_ids.detach().cpu().tolist()
        masks = None if attention_mask is None else attention_mask.detach().cpu().tolist()
        keys = [tuple(t for _, t in self._prompt(ids[r], None if masks is None else masks[r],
                                                 boundaries[r])) for r in range(len(ids))]
        missing = [r for r, k in enumerate(keys) if k not in self.route_cache]
        if missing:
            # Route the uncached prompts on their own (right-padded prefixes; the router
            # reads the boundary, which is causal in the prefix, so this is exact).
            width = max(len(keys[r]) for r in missing)
            device = input_ids.device
            batch = torch.zeros((len(missing), width), dtype=torch.long, device=device)
            att = torch.zeros_like(batch)
            for i, r in enumerate(missing):
                batch[i, :len(keys[r])] = torch.tensor(keys[r], device=device)
                att[i, :len(keys[r])] = 1
            self.router_lm.set_association_prefix_lengths([len(keys[r]) for r in missing])
            with torch.no_grad():
                self.router_lm(input_ids=batch, attention_mask=att, use_cache=False)
            for i, r in enumerate(missing):
                active = self.bank.last_active_fact_indices[i]
                self.route_cache[keys[r]] = int(active[0]) if active else -1
        return [self.route_cache[k] for k in keys]

    def _subject_positions(self, prompt, fact):
        ids = [t for _, t in prompt]
        best = None
        for pattern in self.subject_patterns[fact]:
            width = len(pattern)
            for start in range(len(ids) - width, -1, -1):
                if tuple(ids[start:start + width]) == tuple(pattern):
                    if best is None or start + width > best[1]:
                        best = (start, start + width)
                    break
        if best is None:
            return []
        span = [prompt[k][0] for k in range(best[0], best[1])]
        return span[-1:] if self.write_mode == "subject_last" else span

    # ---------------------------------------------------------------------- editing
    def _hook(self, _module, _args, out):
        if self._plan is None:
            return out
        hidden = _block_output(out)
        mask, facts = self._plan
        if mask.shape[1] != hidden.shape[1]:
            raise RuntimeError("early-write plan width does not match the hidden state")
        active = facts >= 0
        selected = self.rows[facts.clamp_min(0)].to(hidden.dtype)                 # [B, d]
        delta = mask.to(hidden.dtype)[..., None] * selected[:, None, :] \
            * active[:, None, None].to(hidden.dtype)
        return _replace_output(out, hidden + delta)

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("EarlyWriteCausalLM requires input_ids")
        attention_mask = kwargs.get("attention_mask")
        prefix = self._next_prefix_lengths
        self._next_prefix_lengths = None
        boundaries = self._boundaries(input_ids, attention_mask, prefix)
        facts = self._route(input_ids, attention_mask, boundaries)
        self.last_active_fact_indices = [[f] if f >= 0 else [] for f in facts]
        ids = input_ids.detach().cpu().tolist()
        masks = None if attention_mask is None else attention_mask.detach().cpu().tolist()
        plan = torch.zeros(input_ids.shape, dtype=torch.float32)
        for r, f in enumerate(facts):
            if f < 0:
                continue
            prompt = self._prompt(ids[r], None if masks is None else masks[r], boundaries[r])
            positions = self._subject_positions(prompt, f)
            if not positions:
                self.subject_fallbacks += 1
                positions = [boundaries[r] - 1]
            plan[r, positions] = 1.0
        self._plan = (plan.to(input_ids.device),
                      torch.tensor(facts, dtype=torch.long, device=input_ids.device))
        try:
            if prefix is not None:
                self.router_lm.set_association_prefix_lengths(prefix)
            return self.router_lm(input_ids=input_ids, **kwargs)     # router bank adds zeros
        finally:
            self._plan = None

    @torch.no_grad()
    def generate_uncached_fixed_boundary(self, input_ids, attention_mask=None, *,
                                         max_new_tokens=32, eos_token_id=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("batch size 1 only")
        ids = input_ids.clone()
        prompt_length = ids.shape[1]
        eos = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", None)
        eos_ids = (set() if eos is None else {int(e) for e in eos}
                   if isinstance(eos, (list, tuple, set)) else {int(eos)})
        for _ in range(int(max_new_tokens)):
            self.set_association_prefix_lengths([prompt_length])
            logits = self.forward(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  use_cache=False).logits
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) in eos_ids:
                break
        return ids

    # --------------------------------------------------------------------- artifact
    def artifact(self):
        out = dict(self.source_artifact)
        out["rows"] = self.rows.detach().float().cpu()
        out["early_write"] = {"layer": self.write_layer, "mode": self.write_mode}
        return out


def load_early_write_artifact(base_model, artifact):
    spec = artifact["early_write"]
    model = EarlyWriteCausalLM(base_model, artifact, int(spec["layer"]), str(spec["mode"]))
    model.rows.requires_grad_(False)
    return model, model
