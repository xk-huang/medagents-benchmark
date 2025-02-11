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

## Performance Zero-shot

|                Model-Task               |  Mean | Median | Std Dev |  Min  |  Max   | Number of Samples |
|---------------------------------------|-------+--------+---------+-------+--------+-----------------|
|            medqa_gpt-35-turbo           | 25.46 | 25.00  |   4.67  |  8.00 | 67.00  |        1273       |
|               medqa_gpt-4               | 23.40 | 23.00  |   2.54  | 11.00 | 35.00  |        1273       |
|            medqa_gpt-4o-mini            | 24.54 | 24.00  |   3.53  | 15.00 | 41.00  |        1273       |
|               medqa_gpt-4o              | 24.55 | 24.00  |   4.07  | 11.00 | 52.00  |        1273       |
|              medqa_o1-mini              | 21.65 | 22.00  |   2.77  |  5.00 | 37.00  |        1273       |
|              medqa_o3-mini              | 22.72 | 23.00  |   2.42  | 11.00 | 35.00  |        1273       |
|          medqa_QwQ-32B-Preview          | 27.91 | 27.00  |   5.69  |  8.00 | 76.00  |        1273       |
|            medqa_DeepSeek-R1            | 27.97 | 27.00  |   6.10  | 14.00 | 80.00  |        1273       |
|    medqa_Llama-3.3-70B-Instruct-Turbo   | 26.64 | 26.00  |   5.20  |  0.00 | 81.00  |        1273       |
|         medqa_claude-3-5-sonnet         | 29.46 | 29.00  |   8.67  |  0.00 | 84.00  |        1273       |
|          medqa_claude-3-5-haiku         | 25.03 | 26.00  |   7.71  |  4.00 | 70.00  |        1273       |
|          pubmedqa_gpt-35-turbo          | 27.65 | 28.00  |   3.97  |  7.00 | 46.00  |        500        |
|              pubmedqa_gpt-4             | 26.92 | 27.00  |   4.01  |  1.00 | 45.00  |        500        |
|           pubmedqa_gpt-4o-mini          | 26.56 | 26.00  |   3.23  | 17.00 | 45.00  |        500        |
|             pubmedqa_gpt-4o             | 27.49 | 27.00  |   3.90  | 17.00 | 52.00  |        500        |
|             pubmedqa_o1-mini            | 24.62 | 25.00  |   3.10  | 15.00 | 39.00  |        500        |
|             pubmedqa_o3-mini            | 24.09 | 24.00  |   2.44  | 16.00 | 34.00  |        500        |
|         pubmedqa_QwQ-32B-Preview        | 28.62 | 28.00  |   4.62  | 17.00 | 58.00  |        500        |
|           pubmedqa_DeepSeek-R1          | 27.61 | 27.00  |   4.37  | 12.00 | 49.00  |        500        |
|  pubmedqa_Llama-3.3-70B-Instruct-Turbo  | 27.71 | 28.00  |   6.28  |  0.00 | 53.00  |        500        |
|        pubmedqa_claude-3-5-sonnet       | 30.28 | 30.00  |   6.40  |  5.00 | 86.00  |        500        |
|        pubmedqa_claude-3-5-haiku        | 25.40 | 27.00  |   7.69  |  0.00 | 49.00  |        500        |
|           medmcqa_gpt-35-turbo          | 21.79 | 21.00  |   9.34  |  0.00 | 93.00  |        2816       |
|              medmcqa_gpt-4              | 19.27 | 19.00  |   7.04  |  0.00 | 93.00  |        2816       |
|           medmcqa_gpt-4o-mini           | 19.20 | 19.00  |   6.07  |  0.00 | 61.00  |        2816       |
|              medmcqa_gpt-4o             | 19.85 | 20.00  |   7.49  |  0.00 | 93.00  |        2816       |
|             medmcqa_o1-mini             | 17.71 | 18.00  |   5.97  |  0.00 | 43.00  |        2816       |
|             medmcqa_o3-mini             | 19.09 | 19.00  |   5.94  |  0.00 | 72.00  |        2816       |
|         medmcqa_QwQ-32B-Preview         | 27.82 | 25.00  |  15.16  |  0.00 | 100.00 |        2816       |
|           medmcqa_DeepSeek-R1           | 26.01 | 23.00  |  13.54  |  0.00 | 100.00 |        2816       |
|   medmcqa_Llama-3.3-70B-Instruct-Turbo  | 27.10 | 24.00  |  13.51  |  0.00 | 100.00 |        2816       |
|        medmcqa_claude-3-5-sonnet        | 27.42 | 25.50  |  13.29  |  0.00 | 100.00 |        2816       |
|         medmcqa_claude-3-5-haiku        | 26.40 | 24.00  |  12.28  |  0.00 | 96.00  |        2816       |
|         medbullets_gpt-35-turbo         | 26.36 | 26.00  |   4.64  | 14.00 | 46.00  |        308        |
|             medbullets_gpt-4            | 24.15 | 24.00  |   2.26  | 13.00 | 30.00  |        308        |
|          medbullets_gpt-4o-mini         | 25.85 | 25.00  |   3.86  | 14.00 | 41.00  |        308        |
|            medbullets_gpt-4o            | 25.35 | 25.00  |   4.50  | 12.00 | 47.00  |        308        |
|            medbullets_o1-mini           | 21.60 | 21.00  |   2.61  | 15.00 | 29.00  |        308        |
|            medbullets_o3-mini           | 23.59 | 24.00  |   2.31  | 15.00 | 31.00  |        308        |
|        medbullets_QwQ-32B-Preview       | 29.54 | 28.00  |   6.24  | 18.00 | 55.00  |        308        |
|          medbullets_DeepSeek-R1         | 29.44 | 28.00  |   6.98  | 15.00 | 60.00  |        308        |
| medbullets_Llama-3.3-70B-Instruct-Turbo | 26.71 | 26.00  |   5.09  | 15.00 | 45.00  |        308        |
|       medbullets_claude-3-5-sonnet      | 30.95 | 31.00  |   9.76  |  0.00 | 58.00  |        308        |
|       medbullets_claude-3-5-haiku       | 24.68 | 26.00  |   8.94  |  0.00 | 45.00  |        308        |
|            mmlu_gpt-35-turbo            | 25.09 | 24.00  |  10.67  |  0.00 | 95.00  |        1089       |
|                mmlu_gpt-4               | 21.64 | 22.00  |   6.49  |  0.00 | 62.00  |        1089       |
|             mmlu_gpt-4o-mini            | 22.06 | 22.00  |   6.45  |  0.00 | 62.00  |        1089       |
|               mmlu_gpt-4o               | 22.34 | 22.00  |   7.47  |  0.00 | 92.00  |        1089       |
|               mmlu_o1-mini              | 19.84 | 20.00  |   5.50  |  0.00 | 57.00  |        1089       |
|               mmlu_o3-mini              | 20.74 | 21.00  |   5.22  |  0.00 | 67.00  |        1089       |
|           mmlu_QwQ-32B-Preview          | 29.55 | 27.00  |  14.14  |  0.00 | 100.00 |        1089       |
|             mmlu_DeepSeek-R1            | 30.64 | 27.00  |  16.12  |  0.00 | 100.00 |        1089       |
|    mmlu_Llama-3.3-70B-Instruct-Turbo    | 28.11 | 26.00  |  12.83  |  0.00 | 100.00 |        1089       |
|          mmlu_claude-3-5-sonnet         | 30.79 | 29.00  |  14.20  |  0.00 | 95.00  |        1089       |
|          mmlu_claude-3-5-haiku          | 28.88 | 27.00  |  12.80  |  0.00 | 100.00 |        1089       |
|          mmlu-pro_gpt-35-turbo          | 24.48 | 23.00  |   9.14  |  0.00 | 85.00  |        818        |
|              mmlu-pro_gpt-4             | 21.50 | 22.00  |   5.81  |  0.00 | 45.00  |        818        |
|           mmlu-pro_gpt-4o-mini          | 21.99 | 22.00  |   5.89  |  0.00 | 44.00  |        818        |
|             mmlu-pro_gpt-4o             | 22.15 | 22.00  |   6.87  |  0.00 | 92.00  |        818        |
|             mmlu-pro_o1-mini            | 19.89 | 20.00  |   5.24  |  0.00 | 51.00  |        818        |
|             mmlu-pro_o3-mini            | 20.50 | 21.00  |   4.83  |  0.00 | 50.00  |        818        |
|         mmlu-pro_QwQ-32B-Preview        | 30.58 | 27.50  |  14.31  |  0.00 | 100.00 |        818        |
|           mmlu-pro_DeepSeek-R1          | 29.75 | 27.00  |  13.95  |  0.00 | 100.00 |        818        |
|  mmlu-pro_Llama-3.3-70B-Instruct-Turbo  | 28.02 | 26.00  |  12.68  |  0.00 | 100.00 |        818        |
|        mmlu-pro_claude-3-5-sonnet       | 30.93 | 29.00  |  13.29  |  0.00 | 95.00  |        818        |
|        mmlu-pro_claude-3-5-haiku        | 28.86 | 27.00  |  11.65  |  0.00 | 94.00  |        818        |
|          afrimedqa_gpt-35-turbo         | 23.22 | 23.00  |  10.29  |  5.00 | 81.00  |        174        |
|             afrimedqa_gpt-4             | 18.24 | 19.00  |   7.04  |  0.00 | 42.00  |        174        |
|          afrimedqa_gpt-4o-mini          | 18.22 | 19.00  |   5.90  |  0.00 | 32.00  |        174        |
|             afrimedqa_gpt-4o            | 18.60 | 20.00  |   7.23  |  0.00 | 35.00  |        174        |
|            afrimedqa_o1-mini            | 17.33 | 18.00  |   7.11  |  0.00 | 57.00  |        174        |
|            afrimedqa_o3-mini            | 18.53 | 19.00  |   5.89  |  0.00 | 40.00  |        174        |
|        afrimedqa_QwQ-32B-Preview        | 28.52 | 25.00  |  13.78  |  4.00 | 93.00  |        174        |
|          afrimedqa_DeepSeek-R1          | 27.14 | 25.50  |  12.74  |  8.00 | 93.00  |        174        |
|  afrimedqa_Llama-3.3-70B-Instruct-Turbo | 27.20 | 25.00  |  12.84  |  0.00 | 85.00  |        174        |
|       afrimedqa_claude-3-5-sonnet       | 25.71 | 25.00  |  11.74  |  0.00 | 72.00  |        174        |
|        afrimedqa_claude-3-5-haiku       | 26.32 | 25.00  |  10.62  |  0.00 | 79.00  |        174        |
|---------------------------------------|-------+--------+---------+-------+--------+-----------------|