import argparse
import datasets
import lm_dataformat
import re

parser = argparse.ArgumentParser(description="Deduplicate a list of files")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--archive_commit_freq", type=int, default=10_000)
args = parser.parse_args()

dataset = datasets.load_dataset("script.py", data_dir=args.data_dir, split="train")

print(f"n examples before deduplication: {len(dataset)}")


def get_variables(examples):
    """Convert a code string to a list of variables.
    We assume a variable is a 'word' with only alphanumeric characters in it."""
    variables = [" ".join(re.split(r"\W+", text)) for text in examples["text"]]
    return {"variables": variables}


dataset = dataset.map(get_variables, batched=True)

uniques = set(dataset.unique("variables"))
n_uniques = len(uniques)

print(f"found {n_uniques} unique files")


def check_uniques(example, uniques):
    """If an example is in uniques, return True and remove it from uniques."""
    if example["variables"] in uniques:
        uniques.remove(example["variables"])
        return True
    else:
        return False


deduplicated_dataset = dataset.filter(check_uniques, fn_kwargs={"uniques": uniques})

assert len(deduplicated_dataset) == n_uniques

ar = lm_dataformat.Archive(args.output_dir)

for i, example in enumerate(deduplicated_dataset):
    code = example["text"]
    del example["text"]
    del example["variables"]
    ar.add_data(code, meta=example)
    if i > 0 and i % args.archive_commit_freq == 0:
        ar.commit()
ar.commit()
