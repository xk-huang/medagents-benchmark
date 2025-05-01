# Data

Re-Label hard set:

```bash
python data/label_hardset.py
```

This script relabel hard one (w/ gpt4o-mini) for both train and test sets.

Then filter out samples in the original test set:

```bash
datasets=(
afrimedqa
medbullets
medexqa
medmcqa
medqa
medqa_5options
medxpertqa-r
medxpertqa-u
mmlu
mmlu-pro
pubmedqa
)

for dataset in ${datasets[@]}; do
    echo $dataset
done

for dataset in ${datasets[@]}; do
    base_dir=data/$dataset/
    echo "======== Process $base_dir ========"
    echo "======== Test set ========"
    python scripts/get_leftout_data.py \
        --train_dataset $base_dir/test_hard_reimp.jsonl \
        --test_dataset $base_dir/test_hard.jsonl \
        --output_dir $base_dir \
        --output_name test_hard_reimp_leftout

    echo "======== Train set ========"
    python scripts/get_leftout_data.py \
        --train_dataset $base_dir/train_hard_reimp.jsonl \
        --test_dataset $base_dir/test_hard.jsonl \
        --output_dir $base_dir \
        --output_name train_hard_reimp_leftout
    echo "======== Finish $base_dir ========\n"
done
```