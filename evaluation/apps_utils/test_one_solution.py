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
Run solutions from one problem.
"""

import io
import json
import logging
import math
import numpy as np
import os
import pprint
import sys
import apps_utils.testing_util as test_util
import time

# for timing debugging
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm

from typing import List


def print_results(results):
    print(results)
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        res.extend(results[index])
        per_prob_res.append(np.mean(results[index]))
        all_correct.append(np.all(results[index]))
    # res = np.array(res)
    tmp_results = res
    # print(res)
    compile_errors = len(tmp_results[tmp_results == -2])
    runtime_errors = len(tmp_results[tmp_results == -1])
    failures = len(tmp_results[tmp_results == False])
    successes = len(tmp_results[tmp_results == True])
    total_testcases = len(res)
    # if args.debug:
    print(
        f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }"
    )
    print(
        f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}"
    )
    print(f"number of test cases run = {total_testcases}")

    print(
        f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}"
    )
    print(
        f"Strict Accuracy (all test cases passed / total problems) = {successes / total_testcases}"
    )


def eval_and_save_problems(ds, save):
    # test_path = Path(test_loc)
    # problems = list(test_path.glob("*/"))

    # print(len(problems))
    gpt_codes = {}
    gpt_bleu = {}
    gpt_codebleu = {}
    results = {}
    codes_loc = os.path.join(save, f"all_codes.json")
    # if not os.path.exists(codes_loc):
    #     codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    if os.path.exists(codes_loc):
        results_loc = os.path.join(save, f"all_results.json")
    print(codes_loc, results_loc)

    with open(codes_loc, "r") as f:
        gpt_codes = json.load(f)

    # main eval loop
    for index, row in enumerate(tqdm(ds)):
        try:
            # if args.debug:
            #     print(f"\n\nproblem path = {problem}")
            output_str = gpt_codes[str(index)]
            # print("String output: ", output_str)
        except:
            print("CANNOT FIND OUTPUT_STR FOR", io_path)
            continue
        io_path = row["input_output"]  # os.path.join(args.root, problem)

        # with open(os.path.join(prob_path, "solutions.json"), "r") as f:
        #     sols = json.load(f)

        if not os.path.exists(save):
            os.makedirs(save)

        res = []
        for o_idx, o in enumerate(output_str):
            # print(o)
            # if args.debug:
            #     print(f"\nTesting solution {o_idx}")
            curr_res = [-2]
            try:
                curr_res = test_util.run_test(
                    prob_path=io_path, test=o, debug=False  # args.debug
                )
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        # if args.debug:
        
        # print(f"results = {res}")
        # print(len(output_str))
        flattened = [val for sublist in res for val in sublist]
        results[index] = flattened

        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb

                pdb.set_trace()
                print("didn't save problem due to {e}")

    print(
        f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case"
    )
    print_results(results)
