# Hybrid JSON + Frequency Unlearning Scripts

Copy these files into:

```bash
/scratch/yl258/kp759/Unlearning/semantic-unlearning/scripts/
```

Run:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

# 1. Frequency candidates
python scripts/identify_tokens.py \
  --config config/config_3b_instruct_forget05.yaml

# 2. JSON/LLM raw candidates
python scripts/build_llm_forget_bank.py \
  --config config/config_3b_instruct_forget05.yaml \
  --forget-split forget05 \
  --target-model outputs/finetuned_model_3B_instruct \
  --extractor-model Qwen/Qwen2.5-7B-Instruct \
  --extractor-dtype float16 \
  --out-bank outputs/forget_knowledge_bank_llm_forget05_3b_instruct_aggressive.json \
  --out-semantic-tokens outputs/semantic_tokens_json_raw.json

# 3. Hybrid retain-safe merge
python scripts/filter_forget_tokens_retain_tfidf.py \
  --config config/config_3b_instruct_forget05.yaml \
  --tokens-json outputs/semantic_tokens_json_raw.json \
  --freq-json outputs/semantic_tokens_freq.json \
  --out outputs/semantic_tokens.json \
  --max-retain-ratio 0.002 \
  --max-retain-count 5 \
  --max-retain-tfidf 0.01 \
  --min-contrast 20 \
  --max-final-tokens 400

# 4. Erase final hybrid tokens
python scripts/erase_embeddings.py \
  --config config/config_3b_instruct_forget05.yaml \
  --method mean \
  --skip-eval
```
