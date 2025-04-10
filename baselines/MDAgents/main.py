import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from pptree import print_tree
from prettytable import PrettyTable
from multiprocessing import Pool
from utils import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    load_data, create_question, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)
import time

from dotenv import load_dotenv
load_dotenv()

def save_results(results, results_path):
    results = sorted(results, key=lambda x: x['idx'])
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='medexqa')
parser.add_argument('--dataset_dir', type=str, default='../../data/medexqa')
parser.add_argument('--split', type=str, default='test_hard')
parser.add_argument('--model', type=str, default='deepseek-V3')
parser.add_argument('--difficulty', type=str, default='adaptive')
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--num_processes', type=int, default=1)
args = parser.parse_args()

model, client = setup_model(args.model)
test_qa, examplers = load_data(args.dataset_dir, args.split)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

path = os.path.join(os.getcwd(), 'output')
os.makedirs(path, exist_ok=True)
subpath = os.path.join(path, args.dataset)
os.makedirs(subpath, exist_ok=True)

if args.num_samples is None:
    args.num_samples = len(test_qa)

results_path = os.path.join(subpath, f'{args.model}_{args.dataset}_{args.split}_{args.difficulty}.json')
if os.path.exists(results_path):
    with open(results_path, 'r') as file:
        results = json.load(file)
else:
    results = []

# if test.jsonl doesn't have realidx, annotate with 
if 'realidx' not in test_qa[0]:
    test_qa = [{**s, 'realidx': idx} for idx, s in enumerate(test_qa)]

processed_idx = set([r['idx'] for r in results])
new_samples = [s for s in test_qa if s['realidx'] not in processed_idx]
if args.num_samples is not None:
    new_samples = new_samples[:min(args.num_samples, len(new_samples))]

def process_sample(sample):
    time_start = time.time()
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    try:
        question, _ = create_question(sample, args.dataset)
        difficulty, difficulty_usage = determine_difficulty(question, args.difficulty, args.model)
        total_usage['prompt_tokens'] += difficulty_usage['prompt_tokens']
        total_usage['completion_tokens'] += difficulty_usage['completion_tokens']

        print(f"difficulty: {difficulty}")

        if difficulty == 'basic':
            final_decision, final_decision_usage = process_basic_query(question, examplers, args.model, args)
            if len(final_decision['answer']) != 1:
                return None
            total_usage['prompt_tokens'] += final_decision_usage['prompt_tokens']
            total_usage['completion_tokens'] += final_decision_usage['completion_tokens']
        elif difficulty == 'intermediate':
            final_decision, final_decision_usage = process_intermediate_query(question, examplers, args.model, args)
            if len(final_decision['answer']) != 1:
                return None
            total_usage['prompt_tokens'] += final_decision_usage['prompt_tokens']
            total_usage['completion_tokens'] += final_decision_usage['completion_tokens']
        elif difficulty == 'advanced':
            final_decision, final_decision_usage = process_advanced_query(question, args.model, args)
            if len(final_decision['answer']) != 1:
                return None
            total_usage['prompt_tokens'] += final_decision_usage['prompt_tokens']
            total_usage['completion_tokens'] += final_decision_usage['completion_tokens']

        time_end = time.time()
        return {
            'idx': sample['realidx'],
            'realidx': sample['realidx'],
            'question': question,
            'answer_idx': sample['answer_idx'],
            'answer': sample['answer'],
            'options': sample['options'],
            'predicted_answer': final_decision['answer'],
            'difficulty': difficulty,
            'token_usage': total_usage,
            'time_elapsed': time_end - time_start
        }
    except Exception as e:
        print(f"[ERROR] Processing sample {sample['realidx']} failed: {e}")
        return None

try:
    if args.num_processes > 1:
        with Pool(args.num_processes) as p:
            for result in tqdm(p.imap(process_sample, new_samples), total=len(new_samples)):
                if result is not None:
                    results.append(result)
                    save_results(results, results_path)
    else:
        for no, sample in enumerate(tqdm(new_samples)):
            result = process_sample(sample)
            if result is not None:
                results.append(result)
                save_results(results, results_path)
except KeyboardInterrupt:
    print(f"[ERROR] Processing samples interrupted by user")

finally:
    save_results(results, results_path)