# RWKU — Llama-3.2-3B-Instruct

## Development setting

- Target entity: **Stephen King**
- Seed: **0**
- Target entity id: `rwku:1_Stephen_King`
- Neutral/reference answer: `Unknown`
- Sensitive evaluation scope: 48 target-only generated atomic views (14 direct, 34 generated-subject variants)
- External utility source: English Wikipedia contexts drawn from the existing SURE/W1K cache and split into disjoint optimization and held-out pools.
- Status: **post-hoc development setting**. Earlier official RWKU metrics had already been observed before v3.2 method design; this result is not an untouched benchmark claim.

See `Stephen_King_seed0/` for the complete result bundle.
