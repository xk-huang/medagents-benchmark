import os
import json
from typing import Dict, Any, List
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from anthropic import AnthropicBedrock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
import time
from tiktoken import encoding_for_model

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: str

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def split_and_generate(problem: dict, client: Any, model: str = "o3-mini") -> tuple[str, str, str]:
    """Split question and generate completion for memorization analysis.
    
    Args:
        problem: Dictionary containing question data
        client: OpenAI client instance
        model: Model name to use for generation
        
    Returns:
        Tuple of (first_half, second_half, generated_completion)
    """
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    tokenizer = encoding_for_model("gpt-4o-mini")

    if 'q1' in problem and 'q2' in problem and 'generated_text' in problem:
        q1 = problem['q1']
        q2 = problem['q2']
        generated_text = problem['generated_text']
    else:
        # Split question into halves
        tokens = tokenizer.encode(question_text)
        mid_token = len(tokens) // 2
        q1_tokens = tokens[:mid_token]
        q2_tokens = tokens[mid_token:]
        q1 = tokenizer.decode(q1_tokens)
        q2 = tokenizer.decode(q2_tokens)

        try:
            generated_text = ""
            max_retries = 10
            retries = 0
            while not generated_text and retries < max_retries:
                if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                    completion = client.messages.create(
                        model=ANTHROPIC_MODELS[model],
                        messages=[{"role": "assistant", "content": q1.strip()}],
                        temperature=0.0,
                        max_tokens=len(q2_tokens)
                    )
                    try:
                        generated_text = completion.content[0].text
                    except:
                        generated_text = ""
                        q1 = tokenizer.decode(q1_tokens[:-1])
                        q2 = tokenizer.decode([q1_tokens[-1]] + q2_tokens)
                elif model in ["o1-mini", "o3-mini"]:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "assistant", "content": q1.strip()}],
                        # max_tokens=len(q2_tokens)
                    )
                    generated_text = completion.choices[0].message.content.strip()
                    tokenized_generated_text = tokenizer.encode(generated_text)
                    if len(tokenized_generated_text) > len(q2_tokens):
                        generated_text = tokenizer.decode(tokenized_generated_text[:len(q2_tokens)])
                    if not generated_text:
                        q1 = tokenizer.decode(q1_tokens[:-1])
                        q2 = tokenizer.decode([q1_tokens[-1]] + q2_tokens)
                else:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "assistant", "content": q1.strip()}],
                        seed=42,
                        temperature=0.0,
                        max_tokens=len(q2_tokens)
                    )
                    try:
                        generated_text = completion.choices[0].message.content.strip()
                    except:
                        generated_text = ""
                        q1 = tokenizer.decode(q1_tokens[:-1])
                        q2 = tokenizer.decode([q1_tokens[-1]] + q2_tokens)
                retries += 1
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
                
        return {
            **problem,
            'q1': q1,
            'q2': q2,
            'generated_text': generated_text,
            'levenshtein_distance': levenshtein_distance(q2, generated_text)
        }

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_results(results, existing_output_file):
    results = sorted(results, key=lambda x: x['realidx'])
    with open(existing_output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='claude-3-5-haiku')
    parser.add_argument('--dataset_name', default='medqa')
    parser.add_argument('--dataset_dir', default='../../data/medqa/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--output_files_folder', default='./output/')
    parser.add_argument('--num_processes', type=int, default=4)

    args = parser.parse_args()
    
    client_old = AzureOpenAI(
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
    
    if args.model_name in ["o3-mini", "o1-mini"]:
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION_2"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_2"),
            api_key=os.getenv("AZURE_API_KEY_2"),
        )
    elif args.model_name in ["gpt-35-turbo"]:
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_API_VERSION_3"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT_3"),
            api_key=os.getenv("AZURE_API_KEY_3"),
        )
    elif args.model_name in ["Qwen/QwQ-32B", "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.getenv("TOGETHER_API_KEY"),
        )
    elif args.model_name in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
        client = AnthropicBedrock(
            aws_region=os.getenv("AWS_REGION"),
            aws_access_key=os.getenv("AWS_API_KEY"),
            aws_secret_key=os.getenv("AWS_SECRET_KEY"),
        )
    else:
        client = client_old

    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)
    existing_output_file = os.path.join(args.output_files_folder, args.dataset_name, f"{args.model_name.split('/')[-1]}-{args.dataset_name}-{args.split}-prob.json")
    
    if os.path.exists(existing_output_file):
        print(f"Existing output file found: {existing_output_file}")
        with open(existing_output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from existing file.")
    else:
        results = []

    problems = load_jsonl(os.path.join(args.dataset_dir, f"{args.split}.jsonl"))
    for idx, problem in enumerate(problems):
        if 'realidx' not in problem:
            problem['realidx'] = idx

    processed_realidx = {result.get('realidx', None) for result in results}
    problems_to_process = [problem for problem in problems if problem['realidx'] not in processed_realidx]

    print(f"Processing {len(problems_to_process)} problems out of {len(problems)} total problems.")

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = {executor.submit(split_and_generate, problem, client, args.model_name): problem for problem in problems_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems", unit="problem"):
            try:
                result = future.result()
                if result is None:
                    continue
                result.pop('cleanse_cot', None)
                result.pop('predicted_answer_base_direct', None)
                results.append(result)
                save_results(results, existing_output_file)
            except Exception as e:
                print(f"Error processing a problem: {e}")

    save_results(results, existing_output_file)
    print(f"Saved {len(results)} results to {existing_output_file}")