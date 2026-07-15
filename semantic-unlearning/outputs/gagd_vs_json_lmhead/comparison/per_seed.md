# Per-seed GA/GD vs JSON-LMHead-Zero comparison

All rows use the same official MCF split for their seed. Eff/Gen are lower-is-better; Spe is higher-is-better; PPL should remain low/stable.

| Method | Family | Seed | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | reference | 0 | 6.000 | 6.000 | 10.890 | 11.062 |
| json_lmhead_zero_true_restore150 | json_lmhead | 0 | 0.000 | 0.000 | 27.670 | 11.062 |
| base | reference | 1 | 16.000 | 14.000 | 12.560 | 11.062 |
| json_lmhead_zero_true_restore150 | json_lmhead | 1 | 0.000 | 0.000 | 18.480 | 15.188 |
| base | reference | 2 | 14.000 | 17.000 | 8.080 | 11.062 |
| json_lmhead_zero_true_restore150 | json_lmhead | 2 | 2.000 | 2.000 | 23.620 | 11.062 |
| base | reference | 3 | 16.000 | 19.000 | 9.830 | 11.062 |
| json_lmhead_zero_true_restore150 | json_lmhead | 3 | 0.000 | 0.000 | 25.100 | 15.188 |
| base | reference | 4 | 10.000 | 15.000 | 13.610 | 11.062 |
| json_lmhead_zero_true_restore150 | json_lmhead | 4 | 0.000 | 0.000 | 30.050 | 14.688 |
