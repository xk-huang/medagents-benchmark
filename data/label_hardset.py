import openai
import os
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from typing import List, Dict, Tuple, Any

from dotenv import load_dotenv
load_dotenv(override=True)

client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"), 
    api_version=os.getenv("AZURE_API_VERSION")
)
# NOTE(xk): gpt-4o-1120-nofilter-global
"""
AZURE_MODEL_4O='gpt-4o-1120-nofilter-global' AZURE_MODEL_4O_MINI='gpt-4o-mini-20240718-nofilter-global' python data/label_hardset.py 

DEBUG=1 AZURE_MODEL_4O='gpt-4o-0806-nofilter-global' AZURE_MODEL_4O_MINI='gpt-4o-mini-20240718-nofilter-global' python data/label_hardset.py 
"""
model_4o = os.getenv("AZURE_MODEL_4O")
model_4o_mini = os.getenv("AZURE_MODEL_4O_MINI")

if os.getenv("DEBUG"):
    for model in [model_4o, model_4o_mini]:
        print(f"Using model: {model}")
        test_api_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Who are you?"}
            ],
            temperature=0.5
        )
        content = test_api_response.choices[0].message.content
        prompt_tokens = test_api_response.usage.prompt_tokens
        completion_tokens = test_api_response.usage.completion_tokens
        print(f"Response: {content}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion tokens: {completion_tokens}")
        breakpoint()

def inspect_problem(problem: Dict, client: Any, retries: int = 3) -> Tuple[str, Dict]:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = '\n'.join([f"{key}: {value}" for key, value in options.items()])
    messages = [
        {
            "role": "system",
            "content": "You are a detailed and accurate assistant tasked with identifying whether a problem has missing original content."
        },
        {
            "role": "user",
            "content": f"Evaluate the given problem and determine if it explicitly omits original content required to solve it, such as graphs, figures, charts, tables, images, or other visual resources. Follow these rules:\n\n"
                       f"1. Ignore your ability to solve the problem using assumptions, external knowledge, or reasoning.\n"
                       f"2. Focus solely on whether the question refers to or implies missing original content (e.g., 'Refer to the graph above', 'Based on the figure', 'Analyze the table').\n"
                       f"3. Classify the problem as one of the following:\n"
                       f"   - 'Text Only': If all necessary information to solve the problem is present in the text and options.\n"
                       f"   - 'Missing Resource': If the problem refers to or depends on original content (e.g., graphs, figures, charts, tables, images) that is missing.\n\n"
                       f"Problem:\n{question_text}\n\n"
                       f"Options:\n{options_text}\n\n"
                       f"Reply with a detailed explanation, followed by the indicator `###`, and then the classification response ('Text Only' or 'Missing Resource')."
        }
    ]
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                # model="gpt-4o",
                model=model_4o,
                messages=messages,
                temperature=0
            )
            response = completion.choices[0].message.content
            problem['cleanse_cot'] = response
            classification = response.split("###")[-1].strip().lower()
            
            if "only" in classification:
                return "text_only", problem
            elif "miss" in classification:
                return "missing_resource", problem
            else:
                print(f"Warning: Unexpected classification on attempt {attempt + 1}")
                if attempt == retries - 1:
                    return "text_only", problem
                continue
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return "text_only", problem
            continue

def classify_problems_with_llm(problems: List[Dict], client: Any) -> Tuple[List[Dict], List[Dict]]:
    problems_with_missing_resources = []
    problems_with_text_only = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(inspect_problem, problem, client) for problem in problems]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying problems", unit="problem"):
            try:
                classification, problem = future.result()
                if classification == "text_only":
                    problems_with_text_only.append(problem)
                elif classification == "missing_resource":
                    problems_with_missing_resources.append(problem)
            except Exception as e:
                print(f"Error in future result: {e}")

    return problems_with_missing_resources, problems_with_text_only

def solve_and_classify(problem: Dict, client: Any, retries: int = 3) -> Tuple[str, Dict]:
    question_text = problem.get('question', '')
    options = problem.get('options', {})
    options_text = ' '.join([f"({key}) {value}" for key, value in options.items()])
    correct_answer_idx = problem.get('answer_idx', '')
    
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable medical assistant. Provide accurate answers to the medical question based on the given information."
        },
        {
            "role": "user",
            "content": f"Given the following question and options, select the correct answer by returning only the answer index (e.g., 'A', 'B', 'C', or 'D').\n\n"
                       f"Question:\n{question_text}\n\n"
                       f"Options:\n{options_text}\n\n"
                       f"Reply with the answer index only."
        }
    ]
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                # model="gpt-4o-mini",
                model=model_4o_mini,
                messages=messages,
                temperature=0
            )
            predicted_answer = completion.choices[0].message.content.strip()
            problem['predicted_answer_base_direct'] = predicted_answer

            if correct_answer_idx.lower() in predicted_answer.lower():
                return "easy", problem
            else:
                return "hard", problem

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt == retries - 1:
                return "error", problem
            continue

def classify_difficulty(problems: List[Dict], client: Any) -> Tuple[List[Dict], List[Dict]]:
    easy_problems = []
    hard_problems = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(solve_and_classify, problem, client) for problem in problems]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying problems by difficulty", unit="problem"):
            try:
                classification, problem = future.result()
                if classification == "easy":
                    easy_problems.append(problem)
                elif classification == "hard":
                    hard_problems.append(problem)
            except Exception as e:
                print(f"Error in future result: {e}")

    return easy_problems, hard_problems

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def process_file_type(file_type):
    test_files = glob.glob(f'data/**/*{file_type}.jsonl', recursive=True)

    for f in test_files:
        print(f'\n{f}')
        
        # Check if good set exists
        good_file = f.replace(f'{file_type}.jsonl', f'{file_type}_good_reimp.jsonl')
        hard_file = f.replace(f'{file_type}.jsonl', f'{file_type}_hard_reimp.jsonl')
        
        if os.path.exists(good_file):
            print(f'Loading existing good problems from {good_file}')
            good_problems = load_jsonl(good_file)
        else:
            problems = load_jsonl(f)
            _, good_problems = classify_problems_with_llm(problems, client)
            # Save good problems
            with open(good_file, 'w', encoding='utf-8') as fout:
                for problem in good_problems:
                    json.dump(problem, fout, ensure_ascii=False)
                    fout.write('\n')
            print(f'Saved good problems to {good_file}')
        
        print(f'{len(good_problems)} good problems')
        
        # Skip if hard set already exists
        if os.path.exists(hard_file):
            print(f'Hard set already exists at {hard_file}, skipping...')
            continue
            
        # Process good problems to find hard set
        easy_problems, hard_problems = classify_difficulty(good_problems, client)
        print(f'{len(easy_problems)} easy problems')
        print(f'{len(hard_problems)} hard problems')
        
        # Save hard problems
        with open(hard_file, 'w', encoding='utf-8') as fout:
            for problem in hard_problems:
                json.dump(problem, fout, ensure_ascii=False)
                fout.write('\n')
        print(f'Saved hard problems to {hard_file}')


if __name__ == "__main__":
    # file_type_list = ["train", "test"]
    file_type_list = ["test", "train"]
    for file_type in file_type_list:
        process_file_type(file_type)
    
