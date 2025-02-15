import os
import json
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from anthropic import AnthropicBedrock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
import time
import re
import random
from collections import Counter

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: str
    explanation: Optional[str] = None

# Define Anthropic models mapping used for certain models.
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}

# Prompt templates for the MedPrompt process.
MEDPROMPT_GENERATE_PROMPT = (
    "Question:\n{question}\n\nOptions:\n{options}\n\n"
    "Please reason step by step and provide your final answer without any additional text."
)

MEDPROMPT_ENSEMBLE_PROMPT = (
    "You are given a problem:\n{question}\n\n"
    "Below are multiple candidate solutions generated for the problem:\n{solutions}\n\n"
    "Analyze these solutions and decide on the best final answer by returning only the chosen option number from the following options: {option_letters}."
)

# Helper function to call the LLM with a given prompt.
def call_llm(prompt: str, client: Any, model: str) -> tuple:
    if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
        messages = [{"role": "user", "content": prompt}]
        completion = client.messages.create(
            model=ANTHROPIC_MODELS[model],
            messages=messages,
            temperature=0.0,
            max_tokens=4096
        )
        raw_response = completion.content[0].text
    elif model in ["o1-mini", "o3-mini"]:
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        raw_response = completion.choices[0].message.content.strip()
    else:
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=42,
            temperature=0.0
        )
        raw_response = completion.choices[0].message.content.strip()
    return completion, raw_response

def parse_answer(raw_response: str, options: Dict, client_old: Any, explanation: bool = False) -> str:
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
        "strict": True,
    }

    if explanation:
        answer_schema["schema"]["properties"]["explanation"] = {"type": "string"}
        answer_schema["schema"]["required"].append("explanation")

    extraction_prompt = (
        "You are an answer extractor. Extract the answer option from the text below. "
        "Only return the answer as a JSON object following this format: {\"answer_idx\": \"<option>\"" 
        + (", \"explanation\": \"<explanation>\"" if explanation else "") + "}, "
        "where <option> is one of the following: " + ", ".join(options.keys()) + "."
        "\nText:\n" + raw_response
    )
    extraction_messages = [{"role": "user", "content": extraction_prompt}]
    # Parse the response using the schema.
    extraction_completion = client_old.chat.completions.create(
        model='gpt-4o-mini',
        messages=extraction_messages,
        response_format={"type": "json_schema", "json_schema": answer_schema}
    )
    extraction_raw_response = extraction_completion.choices[0].message.content.strip()
    result = AnswerResponse.parse_raw(extraction_raw_response)
    return result.answer_idx, result.explanation if explanation else None

def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
    shuffled_solutions = solutions.copy()
    random.shuffle(shuffled_solutions)
    answer_mapping = {str(i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
    return shuffled_solutions, answer_mapping

# MedPrompt based problem solver.
def run(problem: Dict, client: Any, model: str = "o3-mini", retries: int = 3, num_rounds: int = 3, vote_count: int = 3) -> Dict:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])
    prompt_tokens, completion_tokens = 0, 0
    start_time = time.time()

    candidate_solutions = []
    for i in range(num_rounds):
        generate_prompt = MEDPROMPT_GENERATE_PROMPT.format(
            question=question_text,
            options=options_text,
            option_letters=", ".join(options.keys())
        )
        try:
            completion, raw_response = call_llm(generate_prompt, client, model)
            usage = completion.usage
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens += usage.input_tokens
                completion_tokens += usage.output_tokens
            else:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
            candidate_solutions.append(raw_response)
        except Exception as e:
            print(f"Error during generation LLM call: {e}")
            continue

    counter = Counter()
    for i in range(vote_count):
        shuffled_solutions, answer_mapping = shuffle_answers(candidate_solutions)
        ensemble_solutions_text = ""
        for index, sol in enumerate(shuffled_solutions):
            ensemble_solutions_text += f"{index}: \n{sol}\n\n"

        ensemble_prompt = MEDPROMPT_ENSEMBLE_PROMPT.format(
            question=question_text,
            solutions=ensemble_solutions_text,
            option_letters=", ".join(answer_mapping.keys())
        )
        try:
            completion, raw_response = call_llm(ensemble_prompt, client, model)
            usage = completion.usage
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens += usage.input_tokens
                completion_tokens += usage.output_tokens
            else:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
            ensemble_response = raw_response
        except Exception as e:
            print(f"Error during ensemble LLM call: {e}")
            ensemble_response = candidate_solutions[0] if candidate_solutions else ""

        print(ensemble_response)
        predicted_answer, _ = parse_answer(ensemble_response, answer_mapping, client_old)
        counter[answer_mapping[predicted_answer]] += 1

    predicted_answer = counter.most_common(1)[0][0]
    final_solution = candidate_solutions[predicted_answer]
    predicted_answer, _ = parse_answer(final_solution, options, client_old)
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    problem['predicted_answer'] = predicted_answer
    problem['token_usage'] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    problem['time_elapsed'] = time_elapsed
    return problem

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
    parser.add_argument('--num_rounds', type=int, default=3)
    parser.add_argument('--vote_count', type=int, default=3)
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--output_files_folder', default='./output/')
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--retries', type=int, default=3)

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
    existing_output_file = os.path.join(
        args.output_files_folder,
        args.dataset_name,
        f"{args.model_name.split('/')[-1]}-{args.dataset_name}-{args.split}-medprompt-{args.num_rounds}.json"
    )
    
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
        # Using MedPrompt iterations for problem solving.
        futures = {executor.submit(run, problem, client, args.model_name, args.retries, args.num_rounds, args.vote_count): problem for problem in problems_to_process}
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