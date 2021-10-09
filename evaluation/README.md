# How to Evaluate

## Human Eval

The following steps are required to run the Human Eval step:
1. Ensure you are using python3.7 as required by [human-eval](https://github.com/openai/human-eval). We recommend conda:
```
conda create -n human-eval python=3.7
```
2. Install the dependencies in this folder
```
pip install -r requirements.txt
```
3. Install human-eval by following the instructions on the [human-eval repo](https://github.com/openai/human-eval#usage)


With the following requirements performed you can now run the `evaluation.py` script:
```
python evaluate.py --model_name_or_path=model_name_or_path --human_eval_path=<path/to/human-eval/data/HumanEval.jsonl.gz> --out_path=./model_results
```
So for example if you want to evaluate the EleutherAI GPT Neo 125M