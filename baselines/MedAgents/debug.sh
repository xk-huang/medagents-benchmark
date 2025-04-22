rm -rf ../../outputs/medagents
LOGS_DIR=../../logs
DATA_DIR=../../data
OUTPUT_DIR=../../outputs/medagents
dataset=medqa
model=gpt-4o-1120-nofilter-global
split=test_hard

python main.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --method syn_verif --output_files_folder $OUTPUT_DIR --num_processes 1