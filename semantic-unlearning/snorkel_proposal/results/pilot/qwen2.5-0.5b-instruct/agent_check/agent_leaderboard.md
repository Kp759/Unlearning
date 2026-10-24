# Agent track — Qwen2.5-0.5B, 5 pilot requests, 6 replies (pipeline check, not a result)

Percent of counted episodes; leak columns: lower is better. Counted: forget episodes where the base agent revealed the answer, retain episodes it solved.

| Method | Agent score | A_E (no leak) | A_L (retain) | Leak: thought | Leak: query | Leak: final | Leak any: closed | Leak any: open | No leak by reply 3 | Episodes (forget/retain) |
|---|---|---|---|---|---|---|---|---|---|---|
| ga | 2.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 6/4 |
| noop | 2.0 | 0.0 | 100.0 | 100.0 | 0.0 | 33.3 | 100.0 | 100.0 | 0.0 | 6/4 |
