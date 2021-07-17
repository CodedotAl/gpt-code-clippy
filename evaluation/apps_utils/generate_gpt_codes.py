# MIT License

# Copyright (c) 2021 Dan Hendrycks and contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Run a tranined model to generate Python code.
"""

import io
import json
import logging
import math
import random
import numpy as np
import os
import pprint
import sys
import time
import transformers
import torch

from apps_utils.reindent import run as run_reindent

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False,
        },
    )

    return ret.getvalue()


def generate_prompt(
    test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None
):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"  # \n"
    else:
        _input += "\nUse Call-Based format"  # \n"

    _input += "\nANSWER:\n"

    return _input
