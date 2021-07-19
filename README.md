# GPT-Code-Clippy (GPT-CC)

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

TODO: link to the dataset available on the HuggingFace datasets hub, see: https://github.com/huggingface/datasets/pull/2666

## Models

The GPT-CC models are fine-tuned versions of [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [GPT-Neo](https://github.com/EleutherAI/gpt-neo).

The available models are:

- https://huggingface.co/flax-community/gpt-2-code-clippy
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps-alldata-2
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps-alldata
- https://huggingface.co/flax-community/gpt-code-clippy-1.3B-apps
- https://huggingface.co/flax-community/gpt-code-clippy-125M-1024-filtered
- https://huggingface.co/flax-community/gpt-code-clippy-125M-256
- https://huggingface.co/flax-community/gpt-code-clippy-125M-bs2048-raw
- https://huggingface.co/flax-community/gpt-neo-1.3B-code-clippy-test-1
- https://huggingface.co/flax-community/gpt-neo-1.3B-code-clippy
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-test-1
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-test
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy
- https://huggingface.co/flax-community/gpt-neo-2.7B-code-clippy
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-dedup-filtered-no-resize-2048bs
- https://huggingface.co/flax-community/gpt-neo-125M-code-clippy-dedup-2048

TODO: which is the recommended model?

## Training

Training is done using the training scripts available [here](https://github.com/ncoop57/gpt-code-clippy/tree/camera-ready/training).

TODO: which is the recommended way to train GPT-CC?

## Evaluation

The models are also evaluated on the [APPS](https://github.com/hendrycks/apps) and [HumanEval](https://github.com/openai/human-eval) datasets.

### Human Eval Results

| Model                             |   pass@1    |   pass@2    |   pass@5    |   pass@10   |
| --------------------------------- | :---------: | :---------: | :---------: | :---------: |
| EleutherAI/gpt-neo                |    0.12%    |    0.24%    |    0.61%    |    1.22%    |
| EleutherAI/gpt-neo                |    0.06%    |    0.12%    |    0.30%    |    0.61%    |
| dedup-filtered-no-resize-2048bs   |    0.00%    |    0.00%    |    0.00%    |    0.00%    |
| 1024-filtered                     |    0.00%    |    0.00%    |    0.00%    |    0.00%    |
| dedup-2048                        |    0.00%    |    0.00%    |    0.00%    |    0.00%    |


TODO: evaluation results.

## Demo

A [Visual Studio Code](https://code.visualstudio.com/) which uses the [HuggingFace Inference API](https://api-inference.huggingface.co/docs/python/html/index.html) is available.

TODO: more information about this when complete.

## Further Reading

For more information about GPT-CC, GitHub Copilot, etc, see:

- https://github.blog/2021-06-29-introducing-github-copilot-ai-pair-programmer/

TODO: add more further reading.

## Acknowledgements

- https://github.com/arampacha
- https://github.com/ncoop57
- https://github.com/bentrevett
- https://github.com/arunraja-hub
- https://github.com/reshinthadithyan
- https://github.com/shpotes
- https://github.com/neubig
- https://github.com/Mrinal18

TODO: everyone to add their names here!
