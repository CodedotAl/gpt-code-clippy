# GPT-Code-Clippy (GPT-CC)
**Please refer to our new [GitHub Wiki](https://github.com/ncoop57/gpt-code-clippy/wiki) which documents our efforts in detail in creating the open source version of GitHub  Copilot**

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/ncoop57/gpt-code-clippy/camera-ready/code_clippy_logo.jpg" width="256"/>
    <br>
    Courtesy of the awesome Aimee Trevett!
<p>

## Introduction

GPT-Code-Clippy (GPT-CC) is an open source version of [GitHub Copilot](https://copilot.github.com/), a language model -- based on [GPT-3](https://arxiv.org/abs/2005.14165), called [GPT-Codex](https://arxiv.org/abs/2107.03374) -- that is fine-tuned on publicly available code from GitHub.

## Datasets

The dataset used to train GPT-CC is obtained from [SEART GitHub Search](https://seart-ghs.si.usi.ch/) using the following criteria:

- &gt;10 GitHub stars
- &gt;2 commits
- Must have a licence
- Exclude forks
- Size < 70708 bytes

These repositories are then combined with all of the GitHub repositories contain in [The Pile](https://arxiv.org/abs/2101.00027).

The repositories are then filtered for duplicate files. Filtering is performed by regexing each file in each repository to obtain a list of "variables" (the tokens which only contain alphanumeric characters) and then filtering out any files which contain the same sequence of "variables. The deduplication script is available [here](https://github.com/ncoop57/gpt-code-clippy/tree/camera-ready/data_processing/deduplication).

The final dataset is available [here](https://the-eye.eu/public/AI/training_data/code_clippy_data/code_clippy_dedup_data/). The dataset without the duplicates filtered out is also available [here](https://the-eye.eu/public/AI/training_data/code_clippy_data/code_clippy_dedup_data/).

The datasheet discussing in more detail the construction, usage, and limitation of the dataset can be found [here](https://github.com/ncoop57/datasets/tree/code-clippy/datasets/code_clippy). We hope to get it officially into Huggingface's datasets library [soon](https://github.com/huggingface/datasets/pull/2666)!

## Models

The GPT-CC models are fine-tuned versions of [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [GPT-Neo](https://github.com/EleutherAI/gpt-neo).

The available models can be found [here](https://huggingface.co/models?search=code-clippy)

The ones that perform relatively well (None improve on the standard GPT-Neo 125M model except for APPs specific models and only for the APPs task):
- https://huggingface.co/flax-community/gpt-code-clippy-125M-apps-alldata
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps-alldata-2
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps-alldata
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-dedup-2048

TODO: which is the recommended model?

## Training

Training is done using the training scripts available [here](https://github.com/ncoop57/gpt-code-clippy/tree/camera-ready/training).

For fine-tuning GPTNeo-125M on CodeClippy dataset we used AdamW optimizer (beta1=0.9, beta2=0.95) with GPT3-like learning rate schedule (4k warmup steps from 0 to 5e-5 followed by 50k cosine decay steps to 5e-6), weight decay 0.1 and batch size 1024, sequence length 2048. The choice of relatively large batch size and low LR with long warmup are made to avoid agressive updates and preserve the knowledge contained in pretrained GPTNeo weights.

For fine-tuning GPTNe0-125M on APPS dataset we used AdamW optimizer (beta1=0.9, beta2=0.98) with linear learning rate schedule (800 warmup steps from 0 to peak LR followed by linear decay to 0, a range of value for peak LR was [1e-5; 1e-4]), weight decay 0.1 and batch size 256, sequence length 1024. We trained model for 5 epochs selecting best checkpoint judging by validation loss. The language modelling objective for APPS dataset is modified to backpropagate loss only for the tokens corresponding to code solution (refer to [Hendrycks et al](https://arxiv.org/pdf/2105.09938.pdf) for more details).

For fine-tuning GPTNe0-1.3B on APPS dataset we used [Adafactor optimizer](https://github.com/deepmind/optax/blob/243ed1991b2793e87ab60387f7c3d49d6ab57710/optax/_src/alias.py#L74) with linear learning rate schedule (5k warmup steps from 0 to 2e-5 followed by linear decay to 0), weight decay 0.1 and batch size 24, sequence length 1024. The choice of hyperparameters for 1.3B model is in part determined by hardware limitations. We trained model for 5 epochs selecting best checkpoint judging by validation loss.


TODO: which is the recommended way to train GPT-CC?

## Evaluation

The models are also evaluated on the [APPS](https://github.com/hendrycks/apps) and [HumanEval](https://github.com/openai/human-eval) datasets.

### Human Eval Results

| Model                             |   pass@1    |   pass@2    |   pass@5    |   pass@10   |
| --------------------------------- | :---------: | :---------: | :---------: | :---------: |
| EleutherAI/gpt-neo                |    0.12%    |    0.24%    |    0.61%    |    1.22%    |
| gpt-neo-125M-apps                 |    0.06%    |    0.12%    |    0.30%    |    0.61%    |
| dedup-filtered-no-resize-2048bs   |    0.00%    |    0.00%    |    0.00%    |    0.00%    |
| 1024-filtered                     |    0.00%    |    0.00%    |    0.00%    |    0.00%    |
| dedup-2048                        |    0.00%    |    0.00%    |    0.00%    |    0.00%    |

### APPS Eval Results

Coming soon...

## Demo

A [Visual Studio Code](https://code.visualstudio.com/) which uses the [HuggingFace Inference API](https://api-inference.huggingface.co/docs/python/html/index.html) is available and can be found [here](https://github.com/ncoop57/code-clippy-vscode).

We also have [Huggingface's Space demo](https://huggingface.co/spaces/flax-community/code-clippy-problem-solver) where you can specify and problem in the format of a programming competition question.

TODO: more information about this when complete.

## Further Reading

For more information about GPT-CC, GitHub Copilot, etc, see:

- https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/

TODO: add more further reading.

## Acknowledgements

Special thanks to our contributors!!
- https://github.com/arampacha
- https://github.com/ncoop57
- https://github.com/bentrevett
- https://github.com/arunraja-hub
- https://github.com/reshinthadithyan
- https://github.com/shpotes
- https://github.com/taisazero
- https://github.com/neubig
- https://github.com/Mrinal18
- and everyone else that helped out the project!
