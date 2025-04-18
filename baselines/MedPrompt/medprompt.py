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
import numpy as np
from sklearn.neighbors import NearestNeighbors

load_dotenv()

class AnswerResponse(BaseModel):
    answer_idx: str
    explanation: Optional[str] = None

# Define Anthropic models mapping used for certain models.
ANTHROPIC_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0"
}

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

def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, int]]:
    shuffled_solutions = solutions.copy()
    random.shuffle(shuffled_solutions)
    answer_mapping = {str(i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
    return shuffled_solutions, answer_mapping


def get_embeddings(texts: List[str], client: Any, embedding_model: str) -> List[List[float]]:
    embedding = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    return [embedding.data[0].embedding for _ in texts]

def get_nearest_examples(question: str, nbrs: NearestNeighbors, training_data: List[Dict[str, Any]], k: int,
                         client: Any, embedding_model: str) -> List[Dict[str, Any]]:
    question_embedding = get_embeddings([question], client, embedding_model)[0]
    distances, indices = nbrs.kneighbors([question_embedding])
    return [training_data[i] for i in indices[0]]

def build_few_shot_prompt(question: str,
                          options: Dict[str, str],
                          knn_examples: List[Dict[str, Any]]) -> str:
    """
    Build a few-shot prompt that includes the retrieved KNN examples with their correct answers.
    """
    prompt = "Use these examples to guide your reasoning:\n\n"
    for idx, example in enumerate(knn_examples):
        ex_question = example["question"]
        ex_options = example["options"]
        ex_answer = example["answer"] if "answer" in example else None
        # If stored data has direct 'answer_idx', you could decode it to actual text.
        # For demonstration, let's assume 'answer' is the correct key in the example.
        prompt += f"Example {idx+1}:\n"
        prompt += f"Q: {ex_question}\n"
        if ex_answer is not None:
            prompt += f"A: {ex_answer}\n"
        prompt += "\n"

    prompt += "Now, here is a new question:\n"
    prompt += f"Question:\n{question}\n\nOptions:\n"
    for opt_key, opt_val in options.items():
        prompt += f"{opt_key}. {opt_val}\n"
    prompt += (
        "\nLet's think step by step:\n"
        "1. Consider relevant information from examples\n"
        "2. Compare and eliminate incorrect choices\n"
        "3. Provide the final answer: [X]\n"
    )
    return prompt

def run(problem: Dict,
        client: Any,
        embedding_client: Any,
        model: str = "o3-mini",
        nbrs: NearestNeighbors = None,
        retries: int = 3,
        num_rounds: int = 3,
        vote_count: int = 3,
        training_data: List[Dict[str, Any]] = None,
        k_examples: int = 3,
        embedding_model: str = "miblab-text-embed-small") -> Dict:
    """
    Solve a given problem with a few-shot approach based on KNN examples.
    """
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])
    prompt_tokens, completion_tokens = 0, 0
    start_time = time.time()

    # 1) Retrieve KNN examples if training data is provided
    knn_example_data = []
    if training_data:
        knn_example_data = get_nearest_examples(question_text, nbrs, training_data, k_examples, embedding_client, embedding_model)

    # 2) Build a prompt using the KNN examples
    candidate_solutions = []
    raw_responses = []
    for _ in range(num_rounds):
        # Instead of zero-shot prompt, we use the few-shot prompt with KNN examples
        generate_prompt = build_few_shot_prompt(
            question=question_text,
            options=options,
            knn_examples=knn_example_data
        )
        try:
            completion, raw_response = call_llm(generate_prompt, client, model)
            usage = completion.usage
            # Usage tokens might differ for specific LLM providers
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens += usage.input_tokens
                completion_tokens += usage.output_tokens
            else:
                prompt_tokens += getattr(usage, "prompt_tokens", 0)
                completion_tokens += getattr(usage, "completion_tokens", 0)
            candidate_solutions.append(raw_response)
            raw_responses.append(raw_response)
        except Exception as e:
            print(f"Error during generation LLM call: {e}")
            continue

    counter = Counter()
    ensemble_raw_responses = []

    # Simple ensemble approach for majority voting
    for _ in range(vote_count):
        shuffled_solutions, answer_mapping = shuffle_answers(candidate_solutions)
        ensemble_solutions_text = ""
        for index, sol in enumerate(shuffled_solutions):
            ensemble_solutions_text += f"{index}: \n{sol}\n\n"

        # Combine the solutions in the prompt
        ensemble_prompt = (
            f"You are given a problem:\n{question_text}\n\n"
            "Below are multiple candidate solutions generated for the problem:\n"
            f"{ensemble_solutions_text}\n"
            "Analyze these solutions and decide on the best final answer by returning only the chosen "
            f"option number from the following options: {', '.join(answer_mapping.keys())}."
        )

        try:
            completion, raw_response = call_llm(ensemble_prompt, client, model)
            usage = completion.usage
            if model in ["claude-3-5-sonnet", "claude-3-5-haiku"]:
                prompt_tokens += usage.input_tokens
                completion_tokens += usage.output_tokens
            else:
                prompt_tokens += getattr(usage, "prompt_tokens", 0)
                completion_tokens += getattr(usage, "completion_tokens", 0)
            ensemble_response = raw_response
            ensemble_raw_responses.append(raw_response)
        except Exception as e:
            print(f"Error during ensemble LLM call: {e}")
            ensemble_response = candidate_solutions[0] if candidate_solutions else ""
            ensemble_raw_responses.append(ensemble_response)

        predicted_idx, _ = parse_answer(ensemble_response, answer_mapping, client_old=client)
        counter[answer_mapping[predicted_idx]] += 1

    best_idx = counter.most_common(1)[0][0]
    final_solution = candidate_solutions[best_idx]

    predicted_answer, _ = parse_answer(final_solution, options, client_old=client)
    end_time = time.time()
    time_elapsed = end_time - start_time

    problem['predicted_answer'] = predicted_answer
    problem['token_usage'] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    problem['time_elapsed'] = time_elapsed
    problem['raw_responses'] = raw_responses
    problem['ensemble_raw_responses'] = ensemble_raw_responses
    problem['final_solution'] = final_solution
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
    parser.add_argument('--knn_examples', type=int, default=3,
                        help='Number of nearest neighbors to retrieve')
    parser.add_argument('--embedding_model', type=str, default='miblab-text-embed-small',
                        help='Name of the embedding model to use')

    args = parser.parse_args()

    # Prepare an old and a default client, if needed
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
    elif args.model_name in ["Qwen/QwQ-32B-Preview", "deepseek-ai/DeepSeek-R1",
                             "deepseek-ai/DeepSeek-V3", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
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

    embedding_client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_BASE"),
        api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    )
    train_file = os.path.join(args.dataset_dir, 'train.jsonl')
    if os.path.exists(train_file):
        training_data = load_jsonl(train_file)
    else:
        print("Warning: training file not found. KNN examples won't be used.")
        training_data = []

    if len(training_data) > 100:
        training_data = training_data[:100]

    nbrs = NearestNeighbors(n_neighbors=args.knn_examples, algorithm='brute', metric='cosine')
    nbrs.fit(get_embeddings([item["question"] for item in training_data], embedding_client, args.embedding_model))

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
        futures = {
            executor.submit(
                run,
                problem,
                client,
                embedding_client,
                args.model_name,
                nbrs,
                args.retries,
                args.num_rounds,
                args.vote_count,
                training_data,
                args.knn_examples,
                args.embedding_model
            ): problem
            for problem in problems_to_process
        }
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