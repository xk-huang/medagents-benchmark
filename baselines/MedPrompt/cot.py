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

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: str

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}

def run_cot(question_text: str, options_text: str, client: Any, model: str) -> tuple[Any, str]:
    """Call LLM to get initial answer"""
    if model in ["o1-mini", "o3-mini", "claude-3-5-sonnet", "claude-3-5-haiku"]:
        messages = [{
            "role": "user", 
            "content": (
                "You are a helpful medical expert. Your task is to answer a multi-choice medical question. "
                "Please first think step-by-step and then choose the answer from the provided options. "
                "Organize your output in a json formatted as Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": Str{A/B/C/...}}. "
                "Your responses will be used for research purposes only, so please have a definite answer.\n\n"
                f"Question:\n{question_text}\n\n"
                f"Options:\n{options_text}\n\n"
                "Please think step-by-step and generate your output in json:"
            )
        }]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical expert. Your task is to answer multi-choice medical questions by thinking step-by-step and providing clear explanations."
            },
            {
                "role": "user",
                "content": (
                    "Please answer this medical question by first thinking step-by-step and then choosing from the provided options. "
                    "Organize your output in a json formatted as Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": Str{A/B/C/...}}. "
                    "Your responses will be used for research purposes only, so please have a definite answer.\n\n"
                    f"Question:\n{question_text}\n\n"
                    f"Options:\n{options_text}\n\n"
                    "Please think step-by-step and generate your output in json:"
                )
            }
        ]

    if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
        completion = client.messages.create(
            model=ANTHROPIC_MODELS[model],
            messages=messages,
            temperature=0.0,
            max_tokens=4096
        )
        raw_response = completion.content[0].text
    elif model in ["o1-mini", "o3-mini"]:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        raw_response = completion.choices[0].message.content.strip()
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=42,
            temperature=0.0
        )
        raw_response = completion.choices[0].message.content.strip()

    return completion, raw_response

def parse_answer(raw_response: str, options: Dict, client_old: Any) -> str:
    """Parse LLM response using GPT-4o-mini"""
    answer_schema = {
        "name": "answer_response",
        "schema": {
            "type": "object",
            "properties": {
                "answer_idx": {"type": "string", "enum": list(options.keys())}
            },
            "required": ["answer_idx"],
            "additionalProperties": False
        },
        "strict": True
    }

    extraction_prompt = (
        "You are an answer extractor. Extract the answer option from the text below. "
        "Only return the answer as a JSON object following this format: {\"answer_idx\": \"<option>\"}, "
        "where <option> is one of the following: " + ", ".join(options.keys()) + "."
        "\nText:\n" + raw_response
    )
    extraction_messages = [{"role": "user", "content": extraction_prompt}]
    extraction_completion = client_old.chat.completions.create(
        model="gpt-4o-mini",
        messages=extraction_messages,
        response_format={"type": "json_schema", "json_schema": answer_schema}
    )
    extraction_raw_response = extraction_completion.choices[0].message.content.strip()
    return AnswerResponse.parse_raw(extraction_raw_response).answer_idx.strip()

def run(problem: Dict, client: Any, model: str = "o3-mini", retries: int = 3) -> Dict:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])

    for attempt in range(retries):
        try:
            start_time = time.time()
            
            # First call: get direct answer from LLM
            completion, raw_response = run_cot(question_text, options_text, client, model)
            
            # Second call: parse answer using GPT-4o-mini
            predicted_answer = parse_answer(raw_response, options, client_old)

            # Calculate token usage
            usage_first = completion.usage
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens = usage_first.input_tokens
                completion_tokens = usage_first.output_tokens
            else:
                prompt_tokens = usage_first.prompt_tokens
                completion_tokens = usage_first.completion_tokens

            end_time = time.time()
            time_elapsed = end_time - start_time

            problem['predicted_answer'] = predicted_answer
            problem['token_usage'] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            problem['time_elapsed'] = time_elapsed
            return problem
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return None
            continue

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
    parser.add_argument('--model_name', default='gpt-4o-mini')
    parser.add_argument('--dataset_name', default='medqa')
    parser.add_argument('--dataset_dir', default='../../data/medqa/')
    parser.add_argument('--split', default='test_hard')
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
    elif args.model_name in ["Qwen/QwQ-32B-Preview", "deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
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
    existing_output_file = os.path.join(args.output_files_folder, args.dataset_name, f"{args.model_name.split('/')[-1]}-{args.dataset_name}-{args.split}-cot.json")
    
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
        futures = {executor.submit(run, problem, client, args.model_name): problem for problem in problems_to_process}
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