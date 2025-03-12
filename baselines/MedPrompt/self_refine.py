import os
import json
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from anthropic import AnthropicBedrock
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import argparse
from pydantic import BaseModel
import time
import re

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: Optional[str] = None
    explanation: Optional[str] = None
    review_result: Optional[bool] = None

# Define Anthropic models mapping used for certain models.
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}

# Prompt templates for the self-refine process.
SELF_REFINE_GENERATE_PROMPT = (
    "Question:\n{question}\n\nOptions:\n{options}\n\n"
    "Please reason step by step and provide your final answer (e.g., 'A', 'B', 'C', etc.)."
)

SELF_REFINE_REVIEW_PROMPT = (
    "Given the following problem and solution, please critically evaluate the solution's correctness.\n\n"
    "Problem:\n{problem}\n\nSolution:\n{solution}\n\n"
    "Please reason step by step. If you are more than 95 percent confident that the solution is incorrect, return False and provide feedback on the error. "
    "Otherwise, return True and explain why the solution is correct."
)

SELF_REFINE_REVISE_PROMPT = (
    "Given the following problem, the original solution, and the feedback, please revise the solution to ensure correctness.\n\n"
    "Problem:\n{problem}\n\nOriginal Solution:\n{solution}\n\nFeedback:\n{feedback}\n\n"
    "Please provide the revised solution (e.g., 'A', 'B', 'C', etc.), after reasoning step by step."
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

def parse_answer(raw_response: str, options: Dict, client_old: Any, answer_idx: bool = False, explanation: bool = False, review_result: bool = False) -> str:
    """Parse LLM response using GPT-4o-mini"""
    answer_schema = {
        "name": "answer_response",
        "schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True,
    }

    if answer_idx:
        answer_schema["schema"]["properties"]["answer_idx"] = {"type": "string", "enum": list(options.keys())}
        answer_schema["schema"]["required"].append("answer_idx")

    if explanation:
        answer_schema["schema"]["properties"]["explanation"] = {"type": "string"}
        answer_schema["schema"]["required"].append("explanation")

    if review_result:
        answer_schema["schema"]["properties"]["review_result"] = {"type": "boolean"}
        answer_schema["schema"]["required"].append("review_result")

    extraction_prompt = (
        "You are an answer extractor. Extract the answer option from the text below. "
        "Only return the answer as a JSON object following this format: {\"thinking\": \"<thinking>\", \"answer_idx\": \"<option>\"" + ", \"explanation\": \"<explanation>\"" if explanation else "" + ", \"review_result\": \"<review_result>\"" if review_result else "" + "}, "
        "where <thinking> is the thinking process and <option> is one of the following: " + ", ".join(options.keys()) + "."
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
    return result.answer_idx if answer_idx else None, result.explanation if explanation else None, result.review_result if review_result else None

# Self-refine based problem solver.
def run(problem: Dict, client: Any, model: str = "o3-mini", retries: int = 3, num_rounds: int = 3) -> Dict:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])
    prompt_tokens, completion_tokens = 0, 0
    start_time = time.time()

    # Initial generation step.
    generate_prompt = SELF_REFINE_GENERATE_PROMPT.format(
        question=question_text,
        options=options_text
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
        solution = raw_response
        predicted_answer, _, _ = parse_answer(solution, options, client_old, answer_idx=True)
    except Exception as e:
        print(f"Error during generation LLM call: {e}")
        return None

    problem['refinement_rounds'] = [{
        'predicted_answer': predicted_answer,
        'review_response': raw_response,
        'review_result': None,
        'revised_response': None,
    }]

    # Iterative self-refinement loop.
    for i in range(num_rounds):
        # Review step.
        review_prompt = SELF_REFINE_REVIEW_PROMPT.format(
            problem=question_text,
            solution=solution
        )
        try:
            completion, raw_response = call_llm(review_prompt, client, model)
            usage = completion.usage
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens += usage.input_tokens
                completion_tokens += usage.output_tokens
            else:
                prompt_tokens += usage.prompt_tokens
                completion_tokens += usage.completion_tokens
            review_response = raw_response
        except Exception as e:
            print(f"Error during review LLM call: {e}")
            break

        # Parse the review result; look for True/False in response.
        _, review_explanation, review_result = parse_answer(review_response, options, client_old, answer_idx=False, explanation=True, review_result=True)

        if review_result:
            # If the review is positive, stop refinement.
            predicted_answer, _, _ = parse_answer(solution, options, client_old, answer_idx=True)
            problem['refinement_rounds'].append({
                'predicted_answer': predicted_answer,
                'review_response': review_response,
                'review_result': review_result,
                'revised_response': solution,
            })
            break
        else:
            # Revision step.
            revise_prompt = SELF_REFINE_REVISE_PROMPT.format(
                problem=question_text,
                solution=solution,
                feedback=review_explanation
            )
            try:
                completion, raw_response = call_llm(revise_prompt, client, model)
                usage = completion.usage
                if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                    prompt_tokens += usage.input_tokens
                    completion_tokens += usage.output_tokens
                else:
                    prompt_tokens += usage.prompt_tokens
                    completion_tokens += usage.completion_tokens
                solution = raw_response
            except Exception as e:
                print(f"Error during revision LLM call: {e}")
                break

        predicted_answer, _, _ = parse_answer(solution, options, client_old, answer_idx=True)
        problem['refinement_rounds'].append({
            'predicted_answer': predicted_answer,
            'review_response': review_response,
            'revise_response': raw_response,
            'review_result': review_result,
        })

    # Extract final answer using regex.
    predicted_answer, _, _ = parse_answer(solution, options, client_old, answer_idx=True)
    
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
        f"{args.model_name.split('/')[-1]}-{args.dataset_name}-{args.split}-self_refine-{args.num_rounds}.json"
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
        # Using self-refine iterations for problem solving.
        futures = {executor.submit(run, problem, client, args.model_name, args.retries, args.num_rounds): problem for problem in problems_to_process}
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