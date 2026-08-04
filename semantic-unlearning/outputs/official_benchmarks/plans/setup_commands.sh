#!/usr/bin/env bash
set -euo pipefail

# mcf_zerounlearn_official
test -n "${ZEROUNLEARN_REV:?pin a 40-hex ZeroUnlearn commit}" && git clone https://github.com/XMUDeepLIT/ZeroUnlearn "${OFFICIAL_BENCH_ROOT}/ZeroUnlearn" && git -C "${OFFICIAL_BENCH_ROOT}/ZeroUnlearn" checkout --detach "${ZEROUNLEARN_REV}"

# zsre_zerounlearn_official
test -n "${ZEROUNLEARN_REV:?pin a 40-hex ZeroUnlearn commit}" && git clone https://github.com/XMUDeepLIT/ZeroUnlearn "${OFFICIAL_BENCH_ROOT}/ZeroUnlearn" && git -C "${OFFICIAL_BENCH_ROOT}/ZeroUnlearn" checkout --detach "${ZEROUNLEARN_REV}"

# tofu_forget05
test -n "${OPEN_UNLEARNING_REV:?pin a 40-hex OpenUnlearning commit}" && git clone https://github.com/locuslab/open-unlearning "${OFFICIAL_BENCH_ROOT}/open-unlearning" && git -C "${OFFICIAL_BENCH_ROOT}/open-unlearning" checkout --detach "${OPEN_UNLEARNING_REV}" && ${PYTHON_BIN} -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["TOFU_FULL_MODEL_ID"], revision=os.environ["TOFU_FULL_MODEL_REV"], local_dir=os.environ["TOFU_FULL_MODEL_PATH"])'

# muse_news
test -n "${MUSE_REV:?pin a 40-hex MUSE commit}" && git clone https://github.com/jaechan-repo/muse_bench "${OFFICIAL_BENCH_ROOT}/muse_bench" && git -C "${OFFICIAL_BENCH_ROOT}/muse_bench" checkout --detach "${MUSE_REV}" && cd "${OFFICIAL_BENCH_ROOT}/muse_bench" && ${PYTHON_BIN} load_data.py

# muse_books
test -n "${MUSE_REV:?pin a 40-hex MUSE commit}" && git clone https://github.com/jaechan-repo/muse_bench "${OFFICIAL_BENCH_ROOT}/muse_bench" && git -C "${OFFICIAL_BENCH_ROOT}/muse_bench" checkout --detach "${MUSE_REV}" && cd "${OFFICIAL_BENCH_ROOT}/muse_bench" && ${PYTHON_BIN} load_data.py

# rwku
test -n "${RWKU_REV:?pin a 40-hex RWKU commit}" && git clone https://github.com/jinzhuoran/RWKU "${OFFICIAL_BENCH_ROOT}/RWKU" && git -C "${OFFICIAL_BENCH_ROOT}/RWKU" checkout --detach "${RWKU_REV}" && cd "${OFFICIAL_BENCH_ROOT}/RWKU/process" && ${PYTHON_BIN} data_process.py

# wmdp_bio
test -n "${WMDP_REV:?pin WMDP}" && git clone https://github.com/centerforaisafety/wmdp "${OFFICIAL_BENCH_ROOT}/wmdp" && git -C "${OFFICIAL_BENCH_ROOT}/wmdp" checkout --detach "${WMDP_REV}" && git clone --branch v0.4.2 https://github.com/EleutherAI/lm-evaluation-harness "${OFFICIAL_BENCH_ROOT}/lm-evaluation-harness-v0.4.2"

# wmdp_cyber
test -n "${WMDP_REV:?pin WMDP}" && git clone https://github.com/centerforaisafety/wmdp "${OFFICIAL_BENCH_ROOT}/wmdp" && git -C "${OFFICIAL_BENCH_ROOT}/wmdp" checkout --detach "${WMDP_REV}" && git clone --branch v0.4.2 https://github.com/EleutherAI/lm-evaluation-harness "${OFFICIAL_BENCH_ROOT}/lm-evaluation-harness-v0.4.2"

# wmdp_chem_eval
test -n "${WMDP_REV:?pin WMDP}" && git clone https://github.com/centerforaisafety/wmdp "${OFFICIAL_BENCH_ROOT}/wmdp" && git -C "${OFFICIAL_BENCH_ROOT}/wmdp" checkout --detach "${WMDP_REV}" && git clone --branch v0.4.2 https://github.com/EleutherAI/lm-evaluation-harness "${OFFICIAL_BENCH_ROOT}/lm-evaluation-harness-v0.4.2"

# ugbench_tofu
test -n "${UGBENCH_REV:?pin a 40-hex UGBench commit}" && git clone https://github.com/MaybeLizzy/UGBench "${OFFICIAL_BENCH_ROOT}/UGBench" && git -C "${OFFICIAL_BENCH_ROOT}/UGBench" checkout --detach "${UGBENCH_REV}"

# ugbench_harry_potter
test -n "${UGBENCH_REV:?pin a 40-hex UGBench commit}" && git clone https://github.com/MaybeLizzy/UGBench "${OFFICIAL_BENCH_ROOT}/UGBench" && git -C "${OFFICIAL_BENCH_ROOT}/UGBench" checkout --detach "${UGBENCH_REV}"

# ugbench_zsre
test -n "${UGBENCH_REV:?pin a 40-hex UGBench commit}" && git clone https://github.com/MaybeLizzy/UGBench "${OFFICIAL_BENCH_ROOT}/UGBench" && git -C "${OFFICIAL_BENCH_ROOT}/UGBench" checkout --detach "${UGBENCH_REV}"

# pch_continual
test -n "${FIT_REV:?pin a 40-hex FIT commit}" && git clone https://github.com/XiaoyuXU1/FIT "${OFFICIAL_BENCH_ROOT}/FIT" && git -C "${OFFICIAL_BENCH_ROOT}/FIT" checkout --detach "${FIT_REV}" && ${PYTHON_BIN} -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["PCH_START_MODEL_ID"], revision=os.environ["PCH_START_MODEL_REV"], local_dir=os.environ["PCH_START_MODEL_PATH"])'

# hubble_yago
test -n "${HUBBLE_REV:?pin a 40-hex Hubble commit}" && git clone https://github.com/allegro-lab/hubble "${OFFICIAL_BENCH_ROOT}/hubble" && git -C "${OFFICIAL_BENCH_ROOT}/hubble" checkout --detach "${HUBBLE_REV}" && git clone https://github.com/EleutherAI/lm-evaluation-harness "${OFFICIAL_BENCH_ROOT}/hubble-lm-evaluation-harness" && git -C "${OFFICIAL_BENCH_ROOT}/hubble-lm-evaluation-harness" checkout --detach a7ca04353fe1ff967f6c5b631bc31a10a6943b23 && ${PYTHON_BIN} -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["HUBBLE_YAGO_MODEL_ID"], revision=os.environ["HUBBLE_YAGO_MODEL_REV"], local_dir=os.environ["HUBBLE_YAGO_PERTURBED_MODEL_PATH"])'

# hubble_gutenberg
test -n "${HUBBLE_REV:?pin a 40-hex Hubble commit}" && git clone https://github.com/allegro-lab/hubble "${OFFICIAL_BENCH_ROOT}/hubble" && git -C "${OFFICIAL_BENCH_ROOT}/hubble" checkout --detach "${HUBBLE_REV}" && git clone https://github.com/EleutherAI/lm-evaluation-harness "${OFFICIAL_BENCH_ROOT}/hubble-lm-evaluation-harness" && git -C "${OFFICIAL_BENCH_ROOT}/hubble-lm-evaluation-harness" checkout --detach a7ca04353fe1ff967f6c5b631bc31a10a6943b23 && ${PYTHON_BIN} -c 'from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id=os.environ["HUBBLE_GUTENBERG_MODEL_ID"], revision=os.environ["HUBBLE_GUTENBERG_MODEL_REV"], local_dir=os.environ["HUBBLE_GUTENBERG_PERTURBED_MODEL_PATH"])'
