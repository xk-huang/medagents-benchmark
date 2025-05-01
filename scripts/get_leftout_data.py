"""
We filter those train samples that are in the test set, then save the left-out samples.

python scripts/get_leftout_data.py \
    --train_dataset <train_dataset> \
    --test_dataset <test_dataset> \
    --output_dir <output_dir> \
    --output_name <output_name>

base_dir=data/medbullets/
python scripts/get_leftout_data.py \
    --train_dataset $base_dir/test_hard_reimp.jsonl \
    --test_dataset $base_dir/test_hard.jsonl \
    --output_dir $base_dir \
    --output_name test_hard_reimp_leftout

base_dir=data/medbullets/
python scripts/get_leftout_data.py \
    --train_dataset $base_dir/train_hard_reimp.jsonl \
    --test_dataset $base_dir/test_hard.jsonl \
    --output_dir $base_dir \
    --output_name train_hard_reimp_leftout
"""
import click
from pathlib import Path
import json
from hashlib import md5

@click.command()
@click.option("--train_dataset", type=str, required=True, help="Path to the training dataset")
@click.option("--test_dataset", type=str, required=True, help="Path to the test dataset")
@click.option("--output_dir", type=str, required=True, help="Directory to save the output files")
@click.option("--output_name", type=str, required=True, help="Name of the output file")
def main(
    train_dataset: str,
    test_dataset: str,
    output_dir: str,
    output_name: str,
):
    """
    This script generates a left-out test set from the training dataset.
    """
    train_dataset = Path(train_dataset)
    test_dataset = Path(test_dataset)
    output_dir = Path(output_dir)

    if train_dataset.suffix != ".jsonl":
        raise ValueError(f"Train dataset must be a jsonl file, got {train_dataset.suffix}")
    if test_dataset.suffix != ".jsonl":
        raise ValueError(f"Test dataset must be a jsonl file, got {test_dataset.suffix}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the train jsonl
    train_dataset = read_jsonl_dataset(train_dataset)
    test_dataset = read_jsonl_dataset(test_dataset)

    # get hash of test questions
    test_hashes = set()
    for sample in test_dataset:
        if "question" not in sample:
            raise ValueError(f"Test dataset must contain 'question' field, got {sample}")
        question = sample["question"]
        if isinstance(question, str):
            question = [question]
        for q in question:
            test_hashes.add(md5(q.encode("utf-8")).hexdigest())
    
    # Filter train dataset
    left_out_train_dataset = []
    for idx, sample in enumerate(train_dataset):
        if "question" not in sample:
            raise ValueError(f"Train dataset must contain 'question' field, got {sample}")

        question = sample["question"]
        if isinstance(question, str):
            question = [question]

        for q in question:
            if md5(q.encode("utf-8")).hexdigest() not in test_hashes:
                left_out_train_dataset.append(sample)
    print(f"Filter {len(train_dataset) - len(left_out_train_dataset)} samples from train dataset")
    print(f"Left out {len(left_out_train_dataset)} samples from train dataset")

    # Save the left-out train dataset
    output_file = output_dir / f"{output_name}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(left_out_train_dataset):
            try:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error writing sample {idx}: {e}")
                raise e
    print(f"Saved left-out train dataset to {output_file}")

def read_jsonl_dataset(file_path):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            try:
                sample = json.loads(line)
                dataset.append(sample)
            except Exception as e:
                print(f"Error parsing line {idx}: {line}")
                raise e
    print(f"Loaded {len(dataset)} samples from {file_path}")
    return dataset


if __name__ == "__main__":
    main()