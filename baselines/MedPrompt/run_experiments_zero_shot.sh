#!/bin/bash
#SBATCH --job-name=medagents_experiments
#SBATCH --output=logs/medagents_experiments.log
#SBATCH --error=logs/medagents_experiments.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=scavenge

LOGS_DIR=../../logs/medprompt/zero_shot
DATA_DIR=../../data
OUTPUT_DIR=../../outputs/medprompt/zero_shot

# for model in claude-3-5-haiku claude-3-5-sonnet o1-mini o3-mini gpt-4o-mini gpt-4o Qwen/QwQ-32B deepseek-ai/DeepSeek-R1 deepseek-ai/DeepSeek-V3 meta-llama/Llama-3.3-70B-Instruct-Turbo; do
# for model in Qwen/QwQ-32B; do
# NOTE xk: only 4o is available right now.
for model in gpt-4o-1120-nofilter-global; do
    {
        for dataset in medqa medbullets medmcqa pubmedqa mmlu mmlu-pro medexqa medxpertqa-r medxpertqa-u; do
            mkdir -p $LOGS_DIR/$dataset 
            # for split in test_hard; do
            # for split in test_hard test; do
            for split in test_hard; do
                echo "Running $model on $dataset $split"
                model_filename=$(echo $model | tr '/' '_')
                log_file=$LOGS_DIR/$dataset/${model_filename}_${dataset}_${split}.log
                error_file=$LOGS_DIR/$dataset/${model_filename}_${dataset}_${split}.err
                python zero_shot.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 4 | tee $log_file
                # 2> $error_file
            done
        done
    } &
done
wait