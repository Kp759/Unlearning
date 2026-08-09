# Llama-3.1-8B-Instruct benchmark results

Model snapshot: `0e9e39f249a16976918f6564b8830bc894c89659`

## Available benchmark archives

- `mcf/` — complete. Final 10-seed rank-8 protected sparse LM-head repair result; strict serialized/reloaded Eff=0 and Gen=0 gate passed 10/10.
- `zsre/` — complete under the frozen configured relative utility gates. Eff=0 and Gen=0 on 10/10 seeds, 511 active tokens reduced to zero, with a documented seed-10 absolute-PPL anomaly.
- `tofu/` — pending.

Only results produced by the Llama-3.1-8B-Instruct checkpoint belong in this directory. Each benchmark archive should record its exact model identifier/revision, evaluator protocol, seeds, frozen configuration, provenance, and relevant reproducibility artifacts.
