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

# Define Anthropic models mapping used for certain models.
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}

# Prompt templates for the multipersona debate process.
DEBATE_INITIAL_PROMPT = (
    "You are a {role}. Based on your medical expertise, please analyze the following question and options. "
    "Think step-by-step and provide your reasoning and answer clearly. "
    "Please include distinct sections for your thinking and your answer (e.g. 'A', 'B', 'C', etc.) in your response.\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n"
)

DEBATE_PROMPT = (
    "You are a {role}. Considering the following insights from your peers:\n{context}\n\n"
    "Please update your analysis for the question below. "
    "Think carefully step-by-step and revise your answer accordingly. "
    "Provide your response with clear sections for your updated thinking and updated answer (e.g. 'A', 'B', 'C', etc.).\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n"
)

FINAL_DECISION_PROMPT = (
    "You are a senior medical expert. Considering all the following debate insights and answers:\n\n"
    "{all_thinking}\n\n"
    "{all_answers}\n\n"
    "Please carefully review all the information and provide the final decision. "
    "Offer a detailed rationale and clearly state your final answer, indicating your final reasoning and the chosen option (e.g. 'A', 'B', 'C', etc.). "
    "Note: the final answer should be a single letter corresponding to one of the options.\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n"
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

class AnswerResponse(BaseModel):
    thinking: str
    answer_idx: str

def parse_answer(raw_response: str, options: Dict, client_old: Any) -> str:
    """Parse LLM response using GPT-4o-mini"""
    answer_schema = {
        "name": "answer_response",
        "schema": {
            "type": "object",
            "properties": {
                "thinking": {"type": "string"},
                "answer_idx": {"type": "string", "enum": list(options.keys())}
            },
            "required": ["thinking", "answer_idx"],
            "additionalProperties": False
        },
        "strict": True
    }

    extraction_prompt = (
        "You are an answer extractor. Extract the answer option from the text below. "
        "Only return the answer as a JSON object following this format: {\"thinking\": \"<thinking>\", \"answer_idx\": \"<option>\"}, "
        "where <thinking> is the thinking process and <option> is one of the following: " + ", ".join(options.keys()) + "."
        "\nText:\n" + raw_response
    )
    extraction_messages = [{"role": "user", "content": extraction_prompt}]
    extraction_completion = client_old.chat.completions.create(
        model="gpt-4o-mini",
        messages=extraction_messages,
        response_format={"type": "json_schema", "json_schema": answer_schema}
    )
    extraction_raw_response = extraction_completion.choices[0].message.content.strip()
    result = AnswerResponse.parse_raw(extraction_raw_response)
    return result.thinking, result.answer_idx

# Multipersona debate-based problem solver.
def run(problem: Dict, client: Any, model: str = "o3-mini", retries: int = 3, num_rounds: int = 2) -> Dict:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])
    prompt_tokens, completion_tokens = 0, 0
    start_time = time.time()
    
    # Define debate agent roles (adapted for the medical domain)
    debate_roles = [
        "Innovative Medical Thinker - MD",
        "Critical Medical Analyst - Medical Professor",
        "Clinical Decision Specialist - Medical Researcher"
    ]
    
    rounds_thinking = []  # For storing each agent's thinking per round.
    rounds_answers = []   # For storing each agent's answer per round.
    raw_responses = []    # For storing raw responses from each agent.

    for r in range(num_rounds):
        current_thinking = []
        current_answers = []
        current_raw_responses = []
        for i, role in enumerate(debate_roles):
            if r == 0:
                prompt = DEBATE_INITIAL_PROMPT.format(
                    role=role,
                    question=question_text,
                    options=options_text
                )
            else:
                # Build context using other agents' previous round thinking.
                context_items = []
                for j, other_role in enumerate(debate_roles):
                    if j != i:
                        context_items.append(f"{other_role}'s previous thinking: {rounds_thinking[r-1][j]}")
                context_text = "\n".join(context_items)
                prompt = DEBATE_PROMPT.format(
                    role=role,
                    context=context_text,
                    question=question_text,
                    options=options_text
                )
            try:
                completion, raw_response = call_llm(prompt, client, model)
                usage = completion.usage
                if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                    prompt_tokens += usage.input_tokens
                    completion_tokens += usage.output_tokens
                else:
                    prompt_tokens += usage.prompt_tokens
                    completion_tokens += usage.completion_tokens
            except Exception as e:
                print(f"Error during LLM call for role {role} in round {r+1}: {e}")
                continue
            thinking, answer = parse_answer(raw_response, options, client_old)
            current_thinking.append(thinking)
            current_answers.append(answer)
            current_raw_responses.append(raw_response)
        rounds_thinking.append(current_thinking)
        rounds_answers.append(current_answers)
        raw_responses.append(current_raw_responses)
    
    # Final decision phase using aggregated debate outcomes.
    final_thinking_context = "\n".join([
        f"{debate_roles[i]}'s final thinking: {rounds_thinking[-1][i]}" for i in range(len(debate_roles))
    ])
    final_answers_context = "\n".join([
        f"{debate_roles[i]}'s final answer: {rounds_answers[-1][i]}" for i in range(len(debate_roles))
    ])
    final_prompt = FINAL_DECISION_PROMPT.format(
        question=question_text,
        options=options_text,
        all_thinking=final_thinking_context,
        all_answers=final_answers_context
    )
    try:
        final_completion, final_raw = call_llm(final_prompt, client, model)
        usage_final = final_completion.usage
        if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
            prompt_tokens += usage_final.input_tokens
            completion_tokens += usage_final.output_tokens
        else:
            prompt_tokens += usage_final.prompt_tokens
            completion_tokens += usage_final.completion_tokens
    except Exception as e:
        print(f"Error during final LLM call: {e}")
        return None
    thinking, answer = parse_answer(final_raw, options, client_old)
    predicted_answer = answer
    final_thinking = thinking
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    problem['predicted_answer'] = predicted_answer
    problem['token_usage'] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    problem['time_elapsed'] = time_elapsed
    problem['rounds_thinking'] = rounds_thinking
    problem['rounds_answers'] = rounds_answers
    problem['raw_responses'] = raw_responses
    problem['final_raw_response'] = final_raw
    problem['final_thinking'] = final_thinking
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
    parser.add_argument('--num_rounds', type=int, default=2)
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
        f"{args.model_name.split('/')[-1]}-{args.dataset_name}-{args.split}-multipersona-{args.num_rounds}.json"
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
        # Using 2 rounds for multipersona debate; ignore the old num_solutions parameter.
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