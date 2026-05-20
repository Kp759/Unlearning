# Embedding-only GA/GD unlearning

Copy script:

```bash
cp /mnt/data/embed_ga_gd_scripts/embedding_ga_gd_unlearn.py \
  /scratch/yl258/kp759/Unlearning/semantic-unlearning/scripts/
```

## Step 1: make JSON-TFIDF forget tokens

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

python scripts/filter_forget_tokens_retain_tfidf.py \
  --config config/config_3b_instruct_forget05.yaml \
  --tokens-json outputs/semantic_tokens_json_raw.json \
  --freq-json outputs/NO_FREQ_FILE_USE_JSON_ONLY.json \
  --out outputs/semantic_tokens.json \
  --max-retain-ratio 0.0025 \
  --max-retain-count 7 \
  --max-retain-tfidf 0.012 \
  --min-contrast 5 \
  --max-final-tokens 1000
```

## Step 2: run embedding-only GA/GD

Balanced:

```bash
python scripts/embedding_ga_gd_unlearn.py \
  --config config/config_3b_instruct_forget05.yaml \
  --model-dir outputs/finetuned_model_3B_instruct \
  --forget-token-json outputs/semantic_tokens.json \
  --output-dir outputs/unlearned_model_embed_ga_gd_json_tfidf_1000_balanced \
  --forget-split forget05 \
  --retain-split retain95 \
  --steps 300 \
  --batch-size 2 \
  --retain-batch-size 2 \
  --forget-lr 5e-5 \
  --retain-lr 2e-5 \
  --retain-top-k 5000 \
  --retain-min-count 10 \
  --retain-max-forget-ratio 0.005 \
  --anchor-lambda 0.05 \
  --max-delta-norm 0.35 \
  --forget-loss-weight 1.0 \
  --retain-loss-weight 1.0 \
  --update-lm-head-if-untied
```

More aggressive forgetting:

```bash
python scripts/embedding_ga_gd_unlearn.py \
  --config config/config_3b_instruct_forget05.yaml \
  --model-dir outputs/finetuned_model_3B_instruct \
  --forget-token-json outputs/semantic_tokens.json \
  --output-dir outputs/unlearned_model_embed_ga_gd_json_tfidf_1000_aggressive \
  --forget-split forget05 \
  --retain-split retain95 \
  --steps 500 \
  --batch-size 2 \
  --retain-batch-size 2 \
  --forget-lr 8e-5 \
  --retain-lr 3e-5 \
  --retain-top-k 7000 \
  --retain-min-count 8 \
  --retain-max-forget-ratio 0.007 \
  --anchor-lambda 0.03 \
  --max-delta-norm 0.50 \
  --forget-loss-weight 1.5 \
  --retain-loss-weight 1.0 \
  --update-lm-head-if-untied
```

More retain-safe:

```bash
python scripts/embedding_ga_gd_unlearn.py \
  --config config/config_3b_instruct_forget05.yaml \
  --model-dir outputs/finetuned_model_3B_instruct \
  --forget-token-json outputs/semantic_tokens.json \
  --output-dir outputs/unlearned_model_embed_ga_gd_json_tfidf_1000_safe \
  --forget-split forget05 \
  --retain-split retain95 \
  --steps 200 \
  --batch-size 2 \
  --retain-batch-size 4 \
  --forget-lr 3e-5 \
  --retain-lr 2e-5 \
  --retain-top-k 7000 \
  --retain-min-count 10 \
  --retain-max-forget-ratio 0.004 \
  --anchor-lambda 0.10 \
  --max-delta-norm 0.25 \
  --forget-loss-weight 1.0 \
  --retain-loss-weight 1.5 \
  --update-lm-head-if-untied
```

## Step 3: evaluate

```bash
python scripts/tofu_eval.py \
  --config config/config_3b_instruct_forget05.yaml \
  --model-dir outputs/unlearned_model_embed_ga_gd_json_tfidf_1000_balanced \
  --method embed_ga_gd_json_tfidf_1000_balanced_forget05 \
  --forget-split forget05 \
  --retain-split retain95
```
