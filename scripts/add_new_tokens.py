import json

from fastcore.script import *
from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM

PROMPT = """Hi my name is"""


def fix_model_embds(original_model, new_model, tokenizer):
    embed = new_model.get_input_embeddings()
    embed.weight.data[
        : tokenizer.vocab_size
    ] = original_model.get_input_embeddings().weight.data

    # set new whitespace token embeddings to singular tab token embedding
    tab_embed = embed(tokenizer("\t", return_tensors="pt").input_ids)
    for i in range(tokenizer.vocab_size - 1, tokenizer.vocab_size - 1 + 3):
        embed.weight.data[i, :] = tab_embed.squeeze()


def add_new_tokens_to_model(model_name, new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, additional_special_tokens=new_tokens
    )
    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    new_model = AutoModelForCausalLM.from_pretrained(model_name)

    new_model.resize_token_embeddings(len(tokenizer.vocab))
    fix_model_embds(original_model, new_model, tokenizer)

    return new_model, tokenizer


@call_parse
def main(
    output_path: Param("Path to the directory where the output will be saved", str),
    model_name: Param(
        "Name of the model to add tokens to", str
    ) = "EleutherAI/gpt-neo-125M",
    token_path: Param("Path to new tokens", str) = "new_tokens.json",
):
    with open(token_path, "r") as f:
        new_tokens = json.load(f)
    print("Tokens to be added:", new_tokens)
    new_model, tokenizer = add_new_tokens_to_model(model_name, new_tokens)
    new_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # convert to flax model
    flax_model = FlaxAutoModelForCausalLM.from_pretrained(output_path, from_pt=True)
    flax_model.save_pretrained(output_path)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(PROMPT, return_tensors="jax").input_ids
    out = flax_model.generate(
        input_ids, max_length=100, pad_token_id=tokenizer.pad_token_id
    )
    print(tokenizer.decode(out[0][0]))
