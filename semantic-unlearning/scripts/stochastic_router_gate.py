"""Non-deterministic routing variants of the frozen Router V2 gate.

Router V2 as shipped is a hard argmax over a masked relative-context margin:
the route is a deterministic function of (tokenized prompt, prototypes, tau).
That is what makes Equation (8) -- the inactive-path identity -- exact, and it
is also why the router emits no probability, has no ROC, and offers no
operating point that can be moved after training.

This module keeps every frozen component (subject eligibility, prototypes,
tau, alpha, ambiguity margin) and replaces only the *decision rule*. Four
stochastic regimes are provided, all sharing one seeded torch.Generator so
runs stay reproducible:

  deterministic  current behaviour; kept as the control arm.
  soft           g_i = sigmoid((d_i - tau_i) / T); the selected residual is
                 scaled by g in [0, 1] instead of being switched on. The route
                 index is still argmax. Differentiable, no sampling noise, and
                 it yields a calibratable score.
  bernoulli      fire ~ Bernoulli(g). Randomized decision rule: the expected
                 behaviour equals the soft gate, but each call is a hard
                 intervention, which keeps the residual semantics unchanged.
  gumbel         sample the route index from
                 softmax([(d_1 - tau_1) .. (d_N - tau_N), l_bot] / T)
                 restricted to qualifying candidates, where l_bot is an
                 abstention logit in margin units. This is the only mode that
                 can select a non-argmax association, so it is the one that
                 probes whether the top-1 margin is load-bearing. As T -> 0 it
                 reduces to the deterministic rule.
  mc_query       perturb the addressing query, q <- normalize(q + sigma * xi),
                 then apply the deterministic rule. Running K passes gives a
                 per-prompt activation frequency, i.e. a router-confidence
                 estimate that needs no retraining.

Inactive-path caveat
--------------------
Under `soft`, a request that the deterministic router rejects can still
receive a small nonzero residual, so p_edited(y|x) != p_base(y|x) and the
structural identity in Equation (8) no longer holds as stated. `hard_zero_below`
restores it by flooring the gate to exactly zero below a cutoff; set it to
0.5 to recover exactly the deterministic support set with a soft magnitude on
top. Any paper text claiming exact base-path preservation must state which
mode produced the numbers.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from static_overlap_fact_association_v2_gate import (
    RelationPrototypeAssociationBank,
)
from static_overlap_fact_association_embeddings import AssociationCausalLM


MODES = ("deterministic", "soft", "bernoulli", "gumbel", "mc_query")


class StochasticRelationPrototypeBank(RelationPrototypeAssociationBank):
    """Router V2 with a configurable stochastic decision rule."""

    def __init__(
        self,
        *args,
        mode="deterministic",
        temperature=0.05,
        abstain_logit=0.0,
        query_noise=0.0,
        hard_zero_below=0.0,
        seed=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if str(mode) not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if float(query_noise) < 0:
            raise ValueError("query_noise must be non-negative")
        if not 0.0 <= float(hard_zero_below) <= 1.0:
            raise ValueError("hard_zero_below must lie in [0, 1]")
        self.mode = str(mode)
        self.temperature = float(temperature)
        self.abstain_logit = float(abstain_logit)
        self.query_noise = float(query_noise)
        self.hard_zero_below = float(hard_zero_below)
        self.seed = int(seed)
        self._generator = None
        # Route telemetry that the deterministic bank cannot produce.
        self.last_route_probabilities = []
        self.last_gate_scales = []

    def _gen(self, device):
        if self._generator is None or self._generator.device != torch.device(device):
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def reseed(self, seed):
        """Reset the sampling stream, e.g. between Monte-Carlo repetitions."""
        self.seed = int(seed)
        self._generator = None

    def _gate_scale(self, best_d, best_fact, active):
        """Map the winning margin to a gate magnitude in [0, 1]."""
        tau = self.tau.to(best_d.device)[best_fact]
        g = torch.sigmoid((best_d - tau) / self.temperature)
        g = torch.where(active, g, torch.zeros_like(g))
        if self.hard_zero_below > 0:
            g = torch.where(g < self.hard_zero_below, torch.zeros_like(g), g)
        return g

    def _sample_route(self, ranked, qualifies, device):
        """Gumbel-max over qualifying candidates plus an explicit abstain arm.

        Logits are margins measured from each association's own threshold,
        (d_i - tau_i) / T, with the abstain arm at abstain_logit / T. Centring
        on tau is what makes T -> 0 collapse to the deterministic rule: a
        qualifying candidate has a non-negative centred margin and therefore
        beats abstention in the limit. Scoring raw d against a fixed abstain
        logit would instead abstain on every association whose margins happen
        to be negative, which is most of them, since tau is itself negative.
        `abstain_logit` is thus read in margin units: 0 places abstention
        exactly at the decision boundary, and a negative value makes the router
        harder to silence.
        """
        batch, n_facts = ranked.shape
        tau = self.tau.to(device)[None, :]
        logits = (ranked - tau) / self.temperature
        abstain = torch.full(
            (batch, 1), self.abstain_logit / self.temperature, device=device
        )
        full = torch.cat([logits, abstain], dim=-1)
        # Gumbel(0,1) = -log(-log(U)). Both negations need explicit grouping:
        # `-torch.log(x).clamp_min(e)` parses as `-(log(x).clamp_min(e))`, which
        # clamps a negative log up to +e and then takes log of a negative number.
        uniform = torch.rand(
            full.shape, device=device, generator=self._gen(device)
        ).clamp_min(1e-20)
        noise = -torch.log((-torch.log(uniform)).clamp_min(1e-20))
        choice = (full + noise).argmax(dim=-1)
        abstained = choice == n_facts
        best_fact = choice.clamp_max(n_facts - 1)
        active = (~abstained) & qualifies.any(dim=-1)
        probabilities = torch.softmax(full, dim=-1)
        return best_fact, active, probabilities

    def _hook(self, module, args, output):
        if self._input_ids is None:
            raise RuntimeError("Stochastic association hook fired without input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        device = hidden.device

        prefix_lengths = self._prefix_lengths_for(hidden)
        subject_mask = self._subject_mask(
            self._input_ids,
            prefix_lengths,
            attention_mask=self._attention_mask,
        )
        positions = prefix_lengths - 1
        query = hidden[torch.arange(batch, device=device), positions].float()

        if self.mode == "mc_query" and self.query_noise > 0:
            noise = torch.randn(
                query.shape,
                device=device,
                dtype=query.dtype,
                generator=self._gen(device),
            )
            query = query + self.query_noise * noise
        query = F.normalize(query, dim=-1)

        u, d = self._relation_scores(query)
        alpha = self.alpha.to(device)[None, :]
        tau = self.tau.to(device)[None, :]
        qualifies = subject_mask & (u >= alpha) & (d >= tau)

        ranked = d.masked_fill(~qualifies, float("-inf"))
        qualifying_counts = qualifies.sum(dim=-1)
        probabilities = None

        if self.mode == "gumbel":
            best_fact, active, probabilities = self._sample_route(
                ranked, qualifies, device
            )
            best_d = d[torch.arange(batch, device=device), best_fact]
            separation = torch.full_like(best_d, float("inf"))
            ambiguous = torch.zeros_like(active)
        else:
            best_d, best_fact = ranked.max(dim=-1)
            active = torch.isfinite(best_d)
            if len(self.facts) > 1:
                top2 = ranked.topk(k=2, dim=-1).values
                separation = top2[:, 0] - top2[:, 1]
                ambiguous = (
                    (qualifying_counts > 1)
                    & torch.isfinite(top2[:, 1])
                    & (separation < self.ambiguity_margin)
                )
                active = active & ~ambiguous
            else:
                separation = torch.full_like(best_d, float("inf"))
                ambiguous = torch.zeros_like(active)

        # Gate magnitude. `deterministic` and `gumbel` keep the original unit
        # step so that only the selection rule differs from the control arm.
        if self.mode in ("soft", "mc_query"):
            gate = self._gate_scale(best_d, best_fact, active)
            if self.mode == "mc_query":
                gate = active.to(hidden.dtype)
        elif self.mode == "bernoulli":
            probability = self._gate_scale(best_d, best_fact, active)
            draw = torch.rand(
                probability.shape, device=device, generator=self._gen(device)
            )
            fired = draw < probability
            gate = fired.to(torch.float32)
            active = active & fired
        else:
            gate = active.to(torch.float32)

        gate = gate.to(hidden.dtype)
        rows = self.extra.to(device=device, dtype=hidden.dtype)
        selected = F.embedding(best_fact, rows)
        position_mask = F.one_hot(positions, num_classes=width).to(hidden.dtype)
        delta = (
            position_mask.unsqueeze(-1)
            * selected.unsqueeze(1)
            * gate[:, None, None]
        )
        edited = hidden + delta

        self.calls += 1
        with torch.no_grad():
            fired_mask = gate > 0
            self.active_batch_rows += int(fired_mask.sum())
            self.active_token_positions += int(fired_mask.sum())
            self.last_active_fact_indices = [
                [int(best_fact[i])] if bool(fired_mask[i]) else []
                for i in range(batch)
            ]
            self.last_gate_scales = [float(gate[i]) for i in range(batch)]
            self.last_route_probabilities = (
                probabilities.detach().cpu().tolist()
                if probabilities is not None
                else None
            )
            self.last_route_scores = [
                {
                    "fact_index": int(best_fact[i]) if bool(fired_mask[i]) else None,
                    "u": float(u[i, best_fact[i]]),
                    "d": float(d[i, best_fact[i]]),
                    "gate_scale": float(gate[i]),
                    "qualifying_candidates": int(qualifying_counts[i]),
                    "top1_top2_d_separation": (
                        float(separation[i])
                        if bool(torch.isfinite(separation[i]))
                        else None
                    ),
                    "rejected_as_ambiguous": bool(ambiguous[i]),
                    "mode": self.mode,
                }
                for i in range(batch)
            ]
            for fact_index in best_fact[fired_mask].detach().cpu().tolist():
                self.active_fact_counts[int(fact_index)] += 1

        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def artifact(self):
        record = super().artifact()
        record.update({
            "architecture": "stochastic_relation_prototype_fact_association_bank_v2",
            "routing_mode": self.mode,
            "routing_temperature": self.temperature,
            "routing_abstain_logit": self.abstain_logit,
            "routing_query_noise": self.query_noise,
            "routing_hard_zero_below": self.hard_zero_below,
            "routing_seed": self.seed,
            "deterministic_route": self.mode == "deterministic",
            "inactive_path_identity_exact": self.mode != "soft"
            or self.hard_zero_below > 0,
        })
        return record


def load_stochastic_artifact(base_model, artifact, **overrides):
    """Reload a frozen Router V2 artifact under a stochastic decision rule."""
    settings = {
        "mode": artifact.get("routing_mode", "deterministic"),
        "temperature": artifact.get("routing_temperature", 0.05),
        "abstain_logit": artifact.get("routing_abstain_logit", 0.0),
        "query_noise": artifact.get("routing_query_noise", 0.0),
        "hard_zero_below": artifact.get("routing_hard_zero_below", 0.0),
        "seed": artifact.get("routing_seed", 0),
    }
    settings.update(overrides)
    bank = StochasticRelationPrototypeBank(
        base_model,
        int(artifact["layer"]),
        artifact["positive_prototypes"],
        artifact["negative_prototypes"],
        artifact["alpha"],
        artifact["tau"],
        artifact["subject_patterns"],
        artifact["facts"],
        rows=artifact["rows"],
        ambiguity_margin=float(artifact.get("ambiguity_margin", 0.02)),
        **settings,
    )
    for row in bank.rows:
        row.requires_grad_(False)
    return AssociationCausalLM(base_model, bank), bank


@torch.no_grad()
def monte_carlo_activation(model, bank, tokenizer, prompts, repeats=16,
                           query_noise=0.05, base_seed=0, batch_size=8):
    """Per-prompt activation frequency under query perturbation.

    Returns one record per prompt with the fraction of the `repeats` passes in
    which the router fired, and the modal selected association. A prompt with
    frequency near 0 or 1 is a confident decision; anything in between is a
    boundary case, which is exactly the population the in-sample tau fit
    cannot describe.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    previous_mode, previous_noise = bank.mode, bank.query_noise
    bank.mode, bank.query_noise = "mc_query", float(query_noise)
    counts = [dict() for _ in prompts]
    fires = [0 for _ in prompts]
    try:
        for repeat in range(int(repeats)):
            bank.reseed(int(base_seed) + repeat)
            for start in range(0, len(prompts), int(batch_size)):
                window = prompts[start:start + int(batch_size)]
                encoded = tokenizer(
                    window,
                    padding=True,
                    return_tensors="pt",
                    return_token_type_ids=False,
                ).to(device)
                model(**encoded, use_cache=False)
                for offset, active_ids in enumerate(bank.last_active_fact_indices):
                    index = start + offset
                    if active_ids:
                        fires[index] += 1
                        key = int(active_ids[0])
                        counts[index][key] = counts[index].get(key, 0) + 1
    finally:
        bank.mode, bank.query_noise = previous_mode, previous_noise
        bank.reseed(bank.seed)
    records = []
    for index, prompt in enumerate(prompts):
        tally = counts[index]
        modal = max(tally, key=tally.get) if tally else None
        records.append({
            "prompt": prompt,
            "activation_frequency": fires[index] / float(repeats),
            "modal_fact_index": modal,
            "modal_fraction": (
                tally[modal] / float(repeats) if modal is not None else 0.0
            ),
            "distinct_routes": len(tally),
        })
    return records
