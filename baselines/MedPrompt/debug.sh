LOGS_DIR=../../logs
DATA_DIR=../../data
OUTPUT_DIR=../../outputs/medprompt/debug
model=gpt-4o-1120-nofilter-global
dataset=medqa

mkdir -p $LOGS_DIR/$dataset 
split=test_hard
echo "Running $model on $dataset $split"
model_filename=$(echo $model | tr '/' '_')
log_file=$LOGS_DIR/$dataset/${model_filename}_${dataset}_${split}.log
error_file=$LOGS_DIR/$dataset/${model_filename}_${dataset}_${split}.err

python zero_shot.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1
python few_shot.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1
python cot.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1
python cot_sc.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1
python multi_persona.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1
python self_refine.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --output_files_folder $OUTPUT_DIR --num_processes 1