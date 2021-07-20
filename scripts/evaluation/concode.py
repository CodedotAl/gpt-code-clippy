import pandas as pd

from datasets import load_dataset, load_metric
from fastcore.script import *
from pathlib import Path

bleu = load_metric("sacrebleu")

predictions = ["hello there kenobi", "foo bar foobar"]
references = [
    ["hello there general kenobi"],
    ["foo bar foobar"],  # , "hello there !"],  # , "foo bar foobar"],
]


@call_parse
def main(concode_path: Param("Path to the concode data in CodeXGLUE", str)):
    concode_path = Path(concode_path)
    dataset = load_dataset("json", data_files=str(concode_path / "test.json"))
    print(dataset)
    results = bleu.compute(predictions=predictions, references=references)
    print(list(results.keys()))
    print(round(results["score"], 1))
