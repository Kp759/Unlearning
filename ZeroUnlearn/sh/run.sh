#!/bin/bash
ul_dir=/to/your/path/ZeroUnlearn
model_dir=/to/your/path/models
mkdir -p ${ul_dir}/logs
alg_name=$1
unlearn_num=${2:-50}
retain_num=1000
data_set=(zsre mquake mcf)
model_list=(Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct Qwen3-4B)
use_h=False
dis_retain_and_forget=False
eval_edited_glue=True
eval_base_glue=False
eval_retain=True
edit_layer_num=3
seeds=(1 2 3 4 5 6 7 8 9 10)

# GPU configuration
GPU_MEM_THRESHOLD=5  # GPU memory usage threshold (percentage)

# Get available GPUs
get_available_gpus() {
    local available=()
    for gpu_id in "${GPU_QUEUE[@]}"; do
        if [ "${gpu_status[$gpu_id]}" = "idle" ]; then
            local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "0")
            local mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null || echo "1")
            local mem_percent=$((mem_used * 100 / mem_total))
            if [ "$mem_percent" -lt "$GPU_MEM_THRESHOLD" ]; then
                available+=("$gpu_id")
            fi
        fi
    done
    echo "${available[@]}"
}

# Check completed tasks and release GPUs
check_completed_jobs() {
    for gpu_id in "${!pids[@]}"; do
        local pid="${pids[$gpu_id]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            echo "[$(date '+%H:%M:%S')] GPU ${gpu_id}: Completed ${task_info[$gpu_id]}"
            unset pids[$gpu_id]
            unset task_info[$gpu_id]
            gpu_status[$gpu_id]="idle"
            ((completed++))
        fi
    done
}

# Submit task
submit_task() {
    local gpu_id=$1
    local task=$2
    
    IFS='|' read -r model_name data seed <<< "$task"
    local log_file="${ul_dir}/logs/${alg_name}-${model_name}_${data}_seed${seed}_retain${retain_num}_unlearn${unlearn_num}-$(date +%Y%m%d_%H%M%S).log"
    
    # Build arguments
    local args=(
        --alg_name ${alg_name}
        --model_name ${model_name}
        --hparams_fname ${model_name}.json
        --ds_name ${data}
        --ratio_or_num
        --unlearn_num ${unlearn_num}
        --retain_num ${retain_num}
        --model_path_dir ${model_dir}
        --edit_layer_nums ${edit_layer_num}
        --seed ${seed}
    )
    
    [ "${eval_retain}" == "True" ] && args+=(--eval_retain)
    [ "${eval_edited_glue}" == "True" ] && args+=(--downstream_eval_steps 1)
    [ "${eval_base_glue}" == "True" ] && args+=(--eval_base_glue)
    [ "${use_h}" == "True" ] && args+=(--use_h)
    
    echo "[$(date '+%H:%M:%S')] GPU ${gpu_id}: Starting ${model_name}/${data}/seed${seed}"
    
    CUDA_VISIBLE_DEVICES="${gpu_id}" python experiments/evaluate.py "${args[@]}" \
        >> "${log_file}" 2>&1 &
    
    pids[$gpu_id]=$!
    task_info[$gpu_id]="${model_name}/${data}/seed${seed}"
    gpu_status[$gpu_id]="busy"
}

# ==================== Main Program ====================

# Initialize GPU queue
GPU_QUEUE=($(nvidia-smi --query-gpu=index --format=csv,noheader))
if [ ${#GPU_QUEUE[@]} -eq 0 ]; then
    echo "Error: No GPUs detected."
    exit 1
fi
echo "Available GPUs: ${GPU_QUEUE[@]} (Total: ${#GPU_QUEUE[@]})"

# Initialize GPU status
declare -A gpu_status
declare -A pids
declare -A task_info
for gpu_id in "${GPU_QUEUE[@]}"; do
    gpu_status[$gpu_id]="idle"
done

# Build task queue: model|dataset|seed
declare -a task_queue=()
for model_name in "${model_list[@]}"; do
    for data in "${data_set[@]}"; do
        for seed in "${seeds[@]}"; do
            task_queue+=("${model_name}|${data}|${seed}")
        done
    done
done

total_tasks=${#task_queue[@]}
echo "=========================================="
echo "Task queue: ${#model_list[@]} models x ${#data_set[@]} datasets x ${#seeds[@]} seeds = ${total_tasks} tasks"
echo "=========================================="

task_idx=0
completed=0

# Main loop: poll GPUs and submit tasks
while [ $completed -lt $total_tasks ]; do
    # Check completed tasks
    check_completed_jobs
    
    # Get available GPUs and submit new tasks
    available_gpus=($(get_available_gpus))
    
    for gpu_id in "${available_gpus[@]}"; do
        if [ $task_idx -lt $total_tasks ]; then
            submit_task "$gpu_id" "${task_queue[$task_idx]}"
            ((task_idx++))
        fi
    done
    
    # Display progress
    running=${#pids[@]}
    pending=$((total_tasks - task_idx))
    echo "[$(date '+%H:%M:%S')] Progress: ${completed}/${total_tasks} done, ${running} running, ${pending} pending"
    
    sleep 30
done

echo "=========================================="
echo "All ${total_tasks} tasks completed!"
echo "=========================================="
