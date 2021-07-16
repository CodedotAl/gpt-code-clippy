import argparse
import datasets
import lm_dataformat
import re
import tqdm

parser = argparse.ArgumentParser(description="Deduplicate a list of files")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--archive_commit_freq", type=int, default=10_000)
args = parser.parse_args()

dataset = datasets.load_dataset(
    "script.py", data_dir=args.data_dir, split="train", streaming=True
)


def get_variables(example):
    variables = " ".join(re.split(r"\W+", example["text"]))
    return variables


def get_hash(example):
    variables = " ".join(re.split(r"\W+", example["text"]))
    return hash(variables)


uniques = set()
ar = lm_dataformat.Archive(args.output_dir)
i = 0
for example in tqdm.tqdm(dataset):
    h = get_hash(example)
    if h not in uniques:
        uniques.add(h)
        code = example["text"]
        del example["text"]
        ar.add_data(code, meta=example)
        i += 1
        if i % args.archive_commit_freq == 0:
            ar.commit()
ar.commit()

print(i)
