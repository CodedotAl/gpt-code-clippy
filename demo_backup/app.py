import gradio as gr

from rich.console import Console
from rich.syntax import Syntax
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import webbrowser

# model_name = "flax-community/gpt-code-clippy-1.3B-apps-alldata"
model_name = "flax-community/gpt-code-clippy-125M-apps-alldata"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

console = Console(record=True)


def format_input(question, starter_code=""):
    answer_type = (
        "\
Use Call-Based format\
" if starter_code else "\
Use Standard Input format\
"
    )
    return f"\
QUESTION:\
{question}\
{starter_code}\
{answer_type}\
ANSWER:\
"


def format_outputs(text):
    formatted_text = Syntax(
        text, "python", line_numbers=True, indent_guides=True, word_wrap=True
    )
    console.print(formatted_text)

    return console.export_html(inline_styles=True)


def generate_solution(question, starter_code="", temperature=1.0, num_beams=1):
    prompt = format_input(question, starter_code)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    start = len(input_ids[0])
    output = model.generate(
        input_ids,
        max_length=start + 200,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True,
        temperature=temperature,
        num_beams=int(num_beams),
        no_repeat_ngram_size=None,
        repetition_penalty=None,
        num_return_sequences=None,
    )

    return format_outputs(
        tokenizer.decode(output[0][start:], skip_special_tokens=True).strip()
    )


_EXAMPLES = [
    [
        """
Given a 2D list of size `m * n`. Your task is to find the sum of minimum value in each row.
For Example:
```python
[
  [1, 2, 3, 4, 5],       # minimum value of row is 1
  [5, 6, 7, 8, 9],       # minimum value of row is 5
  [20, 21, 34, 56, 100]  # minimum value of row is 20
]
```
So, the function should return `26` because sum of minimums is as `1 + 5 + 20 = 26`
        """,
        "",
        0.8,
    ],
    [
        """
# Personalized greeting

Create a function that gives a personalized greeting. This function takes two parameters: `name` and `owner`.
        """,
        """
Use conditionals to return the proper message:

case| return
--- | ---
name equals owner | 'Hello boss'
otherwise         | 'Hello guest'
def greet(name, owner):
        """,
        0.8,
    ],
]


inputs = [
    gr.inputs.Textbox(placeholder="Define a problem here...", lines=7),
    gr.inputs.Textbox(placeholder="Provide optional starter code...", lines=3),
    gr.inputs.Slider(0.5, 1.5, 0.1, default=0.8, label="Temperature"),
    gr.inputs.Slider(1, 4, 1, default=1, label="Beam size"),
    gr.inputs.Textbox(lines=1, label="Your GitHub API token")
]

outputs = [gr.outputs.HTML(label="Solution")]
print(outputs)

# adding carbon support

GITHUB_API="https://api.github.com"
API_TOKEN=gr.inputs.Textbox(label="Your GitHub API token")
#form a request URL
url=GITHUB_API+"/gists"

#print headers,parameters,payload
headers={'Authorization':'token %s'%API_TOKEN}
params={'scope':'gist'}
payload={outputs}



res=requests.post(url,headers=headers,params=params,data=json.dumps(payload))


col = st.beta_columns([2, 4])
if col.button("Create a 'carbon' copy"):
    carbon_url='https://carbon.now.sh/'+res.text.split(',')[0].split('/')[-1][:-1]
    webbrowser.open_new(carbon_url)


gr.Interface(
    generate_solution,
    inputs=inputs,
    outputs=outputs,
    title="Code Clippy: Problem Solver",
    examples=_EXAMPLES,
).launch(share=False)
