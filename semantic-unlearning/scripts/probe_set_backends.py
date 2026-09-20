"""Generation backends for the held-out router probe set.

The probe set's only load-bearing property is that it comes from a process
unrelated to `_negative_prompts_for_fact`. Which model supplies that process
is interchangeable, so the backend is pluggable and the choice is recorded in
the manifest rather than baked into the script.

A local open-weights backend is preferred over a hosted API for one reason
that matters to reviewers: it is reproducible. A pinned checkpoint with greedy
decoding and a fixed seed regenerates the identical probe set years later,
whereas a hosted endpoint drifts under you and cannot be rerun. The frozen
probe-set JSON remains the artifact the paper cites either way, but with a
local backend the provenance claim is checkable rather than asserted.

Two guards, both deliberate:

  Same-model refusal. Generating probes with the model under evaluation
  reintroduces the shared-bias problem the probe set exists to eliminate: the
  probes inherit whatever phrasings that model finds natural, which is
  correlated with what it finds easy. The backend refuses when the generator
  path resolves to the evaluated model unless explicitly overridden.

  Line format, not JSON. Small instruct models emit malformed JSON often
  enough to corrupt a generation run silently. One sentence per numbered line
  parses robustly, degrades gracefully, and is easy to eyeball.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

_NUMBERED = re.compile(r"^\s*(?:\d+[\.\):]|[-*•])\s*(.+?)\s*$")
_FENCE = re.compile(r"^\s*```(?:json|text)?\s*$|^\s*```\s*$")


def parse_lines(text, expected=None):
    """Parse a model response into a list of probe sentences.

    Accepts numbered lists, bulleted lists, bare lines, and a JSON array, in
    that order of preference. Anything that survives is stripped of enclosing
    quotes and surrounding markdown fences.
    """
    if text is None:
        return []
    # Strip markdown fences before anything else, or a fenced JSON array is
    # not recognised as JSON and survives as a single literal "probe".
    body = "\n".join(
        line for line in text.strip().splitlines() if not _FENCE.match(line)
    )
    stripped = body.strip()
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, list):
                return [str(x).strip() for x in payload if str(x).strip()]
        except ValueError:
            pass
    items = []
    for raw in stripped.splitlines():
        if _FENCE.match(raw):
            continue
        line = raw.strip()
        if not line:
            continue
        match = _NUMBERED.match(line)
        candidate = match.group(1) if match else line
        candidate = candidate.strip().strip('"').strip("'").strip()
        # Drop conversational scaffolding a small instruct model prepends.
        if candidate.endswith(":") and len(candidate.split()) <= 8:
            continue
        if candidate:
            items.append(candidate)
    if expected is not None and len(items) > int(expected):
        items = items[: int(expected)]
    return items


class GeminiBackend:
    """Hosted generation. Convenient, not reproducible."""

    name = "gemini"

    def __init__(self, model="gemini-2.5-pro", temperature=1.0, retries=4,
                 timeout=90, sleep=0.0):
        self.model = str(model)
        self.temperature = float(temperature)
        self.retries = int(retries)
        self.timeout = int(timeout)
        self.sleep = float(sleep)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise SystemExit("GEMINI_API_KEY is not set")

    def describe(self):
        return {
            "backend": self.name,
            "generator_model": self.model,
            "temperature": self.temperature,
            "reproducible": False,
            "reproducibility_note": (
                "Hosted endpoint; the served checkpoint may change without "
                "notice. The frozen probe-set JSON is the citable artifact."
            ),
        }

    def generate(self, prompts, expected=None):
        results = []
        for prompt in prompts:
            results.append(parse_lines(self._one(prompt), expected=expected))
            if self.sleep:
                time.sleep(self.sleep)
        return results

    def _one(self, prompt):
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }).encode("utf-8")
        url = GEMINI_ENDPOINT.format(model=self.model)
        last = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return payload["candidates"][0]["content"]["parts"][0]["text"]
            except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
                last = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini request failed after {self.retries} tries: {last}")


class LocalBackend:
    """Open-weights generation through transformers. Reproducible and offline.

    Greedy decoding by default so the probe set is a deterministic function of
    (checkpoint, prompt template, class list). Pass a nonzero temperature only
    if diversity matters more than exact reproducibility; the seed is recorded
    either way.
    """

    name = "local"

    def __init__(self, model_path, device="cuda", dtype="bfloat16",
                 temperature=0.0, top_p=0.95, max_new_tokens=512, seed=0,
                 batch_size=4, local_files_only=False,
                 evaluated_model_path=None, allow_same_model=False):
        # Check before importing or loading anything: this guard must not cost
        # the user a 14B checkpoint load before it fires.
        if evaluated_model_path and not allow_same_model:
            generator = Path(str(model_path)).resolve()
            evaluated = Path(str(evaluated_model_path)).resolve()
            if generator == evaluated:
                raise SystemExit(
                    "Refusing to generate probes with the model under "
                    "evaluation: the probe set would inherit that model's own "
                    "phrasing preferences, which is the shared-generator bias "
                    "this probe set exists to remove. Use a different family "
                    "or size, or pass --allow-same-model to override."
                )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_path = str(model_path)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only batched generation needs left padding or the shorter
        # prompts in a batch decode from padding tokens.
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=local_files_only,
            torch_dtype=getattr(torch, str(dtype)),
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device

    def describe(self):
        return {
            "backend": self.name,
            "generator_model": self.model_path,
            "generator_model_sha256_config": _config_hash(self.model_path),
            "temperature": self.temperature,
            "top_p": self.top_p if self.temperature > 0 else None,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "decoding": "greedy" if self.temperature == 0 else "sampling",
            "reproducible": self.temperature == 0,
            "reproducibility_note": (
                "Pinned local checkpoint; greedy decoding makes the probe set a "
                "deterministic function of checkpoint and prompt template."
            ),
        }

    def _chat(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template") and (
            self.tokenizer.chat_template
        ):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(self, prompts, expected=None):
        torch = self.torch
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            window = [self._chat(p) for p in prompts[start:start + self.batch_size]]
            encoded = self.tokenizer(
                window,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
            torch.manual_seed(self.seed + start)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for row, sequence in enumerate(generated):
                completion = sequence[encoded["input_ids"].shape[1]:]
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                outputs.append(parse_lines(text, expected=expected))
        return outputs


def _config_hash(model_path):
    """Hash the checkpoint config so the manifest pins something verifiable."""
    config = Path(model_path) / "config.json"
    if not config.is_file():
        return None
    return hashlib.sha256(config.read_bytes()).hexdigest()


def build_backend(args, evaluated_model_path=None):
    if args.backend == "gemini":
        return GeminiBackend(
            model=args.generator_model,
            temperature=args.temperature,
            sleep=getattr(args, "sleep", 0.0),
        )
    if args.backend == "local":
        return LocalBackend(
            model_path=args.generator_model,
            device=getattr(args, "generator_device", "cuda"),
            dtype=getattr(args, "generator_dtype", "bfloat16"),
            temperature=args.temperature,
            max_new_tokens=getattr(args, "max_new_tokens", 512),
            seed=getattr(args, "seed", 0),
            batch_size=getattr(args, "generator_batch_size", 4),
            local_files_only=getattr(args, "local_files_only", False),
            evaluated_model_path=evaluated_model_path,
            allow_same_model=getattr(args, "allow_same_model", False),
        )
    raise SystemExit(f"Unknown backend: {args.backend}")
