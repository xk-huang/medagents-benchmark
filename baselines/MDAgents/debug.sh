rm -rf ../../outputs/mdagents/debug
LOGS_DIR=../../logs
DATA_DIR=../../data
OUTPUT_DIR=../../outputs/mdagents/debug
dataset=medqa
model=gpt-4o-1120-nofilter-global
split=test_hard
difficulty=adaptive

python main.py --dataset $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --difficulty $difficulty --output_dir $OUTPUT_DIR --num_samples 1