# MedAgents-Benchmark

This repository contains the evaluation benchmark for medical question-answering agents.

## Installation

Please install the dependencies using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

Put all the environment variables in the `.env` file.

## Running Experiments

To run the baseline experiments:

1. Navigate to the respective baseline directory:
   - `baselines/MDAgents/`
   - `baselines/MedAgents/` 
   - `baselines/MedPrompt/`

2. Execute the experiment script:
   ```bash
   ./run_experiments_all.sh
   ```

3. For analyzing results and calculating error/success metrics, refer to `misc.ipynb`

## Dataset Statistics

The benchmark focuses on challenging medical questions, specifically selecting questions where models achieve less than 50% accuracy. The hard question distribution across tasks is:

| Task       | Number of Hard Questions |
|------------|-------------------------|
| medqa      | 100                     |
| pubmedqa   | 100                     |
| medmcqa    | 100                     |
| medbullets | 89                      |
| mmlu       | 73                      |
| mmlu-pro   | 100                     |
| afrimedqa  | 32                      |

All agent evaluations are conducted on this test_hard subset.
