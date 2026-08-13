# MCF locked protocol: fixed-SVD vs sparse LoRA r=1

Recorded: 2026-08-12  
Model: `meta-llama/Llama-3.2-3B-Instruct`  
Seeds: 1-10  
Slurm job: `1174002`  
Protocol: `zerounlearn_data_access_forget_only_locked_probes`

The LoRA run reused the exact Stage-1 checkpoints from `outputs/mcf_zerounlearn_forget_only_locked_3b`; only the Stage-2 parameterization changed.

| Architecture | Eff ↓ | Gen ↓ | Spe ↑ | Spe_success ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed SVD | 0.0000 ± 0.0000 | 4.0000 ± 3.6332 | 27.7110 ± 3.6742 | 96.3000 ± 1.8639 | 11.5500 ± 0.6771 |
| Sparse LoRA r=1 | 0.0000 ± 0.0000 | 4.0000 ± 3.3466 | 27.7040 ± 3.6780 | 96.3000 ± 1.7894 | 11.5500 ± 0.6771 |

Population SD (`ddof=0`) is shown above.

## Paired LoRA - fixed deltas

| Metric | Mean paired delta | Population SD |
|---|---:|---:|
| Eff | +0.0000 | 0.0000 |
| Gen | +0.0000 | 0.4472 |
| Spe | -0.0070 | 0.0155 |
| Spe_success | -0.0000 | 0.1549 |
| PPL | +0.0000 | 0.0000 |

## Repair complexity

Repair activated on the same five seeds for both methods: **1, 2, 5, 8, 10**. Seeds **3, 4, 6, 7, 9** required no Stage-2 repair.

For each repaired seed:

- Fixed SVD: 2 selected LM-head rows, actual rank 1, **2 trainable Stage-2 coefficients**.
- Sparse LoRA r=1: 2 selected LM-head rows, rank 1, **3074 trainable Stage-2 parameters**.
- LoRA therefore uses **1537× more Stage-2 trainable parameters** for essentially identical performance.

## Conclusion

The sparse-LoRA parameterization does not provide a meaningful performance benefit over the fixed-SVD repair. Keep **fixed-SVD selected-row repair** as the primary SURE-LM Stage-2 architecture and retain LoRA as a negative/efficiency ablation.

Local artifacts:

- Fixed root: `outputs/mcf_zerounlearn_forget_only_locked_3b`
- LoRA root: `outputs/mcf_zerounlearn_forget_only_locked_3b_lora_r1`
- Comparison JSON: `outputs/mcf_locked_fixed_vs_lora_r1.json`
- Comparison Markdown: `outputs/mcf_locked_fixed_vs_lora_r1.md`

This record is append-only and does **not** supersede the existing `BEST_ACCEPTED` fixed-SVD MCF record.
