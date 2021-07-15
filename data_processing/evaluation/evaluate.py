import json
import torch
import pandas as pd

# import apps.eval.reident

from apps_utils.generate_gpt_codes import generate_prompt
from apps_utils.test_one_solution import eval_and_save_problems
from datasets import load_dataset, load_metric
from fastcore.script import *
from human_eval.data import write_jsonl, read_problems
from pathlib import Path
from metrics.extrinsic_eval import compute_metrics
from subprocess import check_output
from transformers import AutoTokenizer, AutoModelWithLMHead

bleu = load_metric("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelWithLMHead.from_pretrained(
    "/home/nathan/gpt-code-clippy/data/APPS/models/1.5B"
)


def generate_text(prompt):
    # print(prompt)
    input_ids = torch.LongTensor(tokenizer.encode(prompt, verbose=False)).unsqueeze(
        0
    )  # .cuda()
    output_ids = model.generate(
        input_ids,
        num_beams=2,
        early_stopping=True,
        max_length=1024 - len(input_ids),
    )
    output_str = tokenizer.decode(output_ids[0])
    return output_str
    # # "a", "=", "b", "\n", "y", "=", "a", "+", "1"
    # return "a = b \n y = a + 1"


def _eval_concode(path):
    # TODO: format input to model same as App and OpenAI HumanEval datasets are formatted
    data = load_dataset("json", data_files=str(path / "test.json"))["train"]
    predictions = [[]]
    references = []
    for example in data:
        output = generate_text(example["nl"])
        predictions[0].append(output.split(" "))
        references.append(example["code"].split(" "))
    results = compute_metrics(predictions, references)
    print(f"Bleu score for Concode dataset: {results}")


def _eval_apps(path):
    gpt_codes = {}
    prob_paths = sorted(path.glob("*/"))
    # map prob_paths to strings and save as a json file
    str_paths = [str(p) for p in prob_paths]
    with open(path / "test.json", "w") as f:
        json.dump(str_paths, f)
    for index, prob_path in enumerate(prob_paths[:2]):
        test_case_path = prob_path / "input_output.json"
        prompt_path = prob_path / "question.txt"
        starter_path = prob_path / "starter_code.py"
        solutions_path = prob_path / "solutions.json"
        if not starter_path.exists():
            starter_path = None
        if not test_case_path.exists() or not prompt_path.exists():
            continue
        prompt = generate_prompt(
            Args(),
            test_case_path,
            prompt_path,
            solutions_path,
            tokenizer,
            starter_path=starter_path,
        )
        output = generate_text(prompt)
        print(output)
        # print(output)
        gpt_codes[index] = output
        # print(output)

    with open(path.parent / "all_codes.json", "w") as f:
        json.dump(gpt_codes, f)

    eval_and_save_problems(path, path.parent)

    # execute bash command to run eval script
    # results = check_output(
    #     [
    #         # python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir --print_results
    #         "python",
    #         "./apps_utils/test_one_solution.py",
    #         "-t",
    #         str(path),
    #         "--save",
    #         str(path.parent),
    #         "--print_results",
    #     ]
    # ).decode("utf-8")


#     test_case_path = os.path.join(prob_path, "input_output.json")
#     prompt_path = os.path.join(prob_path, "question.txt")
#     starter_path = os.path.join(prob_path, "starter_code.py")
#     solutions_path = os.path.join(prob_path, "solutions.json")
#  generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None)


def _eval_human_eval(path):
    problems = read_problems()
    num_samples_per_task = 1
    samples = [
        dict(
            task_id=task_id,
            completion=generate_text(problems[task_id]["prompt"]),
        )
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl("human_eval.jsonl", samples)
    # execute bash command to run eval script
    results = check_output(
        [
            "python",
            path / "evaluate_functional_correctness.py",
            "human_eval.jsonl",
        ]
    ).decode("utf-8")

    print(results)


@call_parse
def main(
    concode_path: Param("Path to the concode data in CodeXGLUE", str),
    apps_path: Param("Path to the the App dataset", str),
    human_eval_path: Param("Path to the human eval dataset", str),
):
    concode_path = Path(concode_path)
    apps_path = Path(apps_path)
    human_eval_path = Path(human_eval_path)
    # _eval_concode(concode_path)
    # _eval_human_eval(human_eval_path)
    _eval_apps(apps_path)
    # dataset = load_dataset("json", data_files=str(concode_path / "test.json"))
    # print(dataset)
    # results = bleu.compute(predictions=predictions, references=references)
    # print(list(results.keys()))
    # print(round(results["score"], 1))


# problems = read_problems()
# print(problems)
# num_samples_per_task = 200
# samples = [
#     dict(
#         task_id=task_id,
#         completion=generate_text(problems[task_id]["prompt"]),
#     )
#     for task_id in problems[:1]
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("human_eval.jsonl", samples)
