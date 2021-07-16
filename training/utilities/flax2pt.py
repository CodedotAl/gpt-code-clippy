#!/usr/bin/env python
import argparse

from transformers import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", help="Path to directory containing config.json and flax_model.msgpack")

args = parser.parse_args()

print(f"Loading flax model from {args.model_dir}...")
model = AutoModel.from_pretrained(args.model_dir, from_flax=True)
print(f"Saving pytorch model...")
model.save_pretrained(args.model_dir, save_config=False)
