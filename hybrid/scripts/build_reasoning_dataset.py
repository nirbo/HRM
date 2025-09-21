#!/usr/bin/env python  # shebang for direct execution
"""Generate reasoning-focused prompts/responses from select Hugging Face datasets."""  # module docstring describing purpose

import argparse  # parse CLI arguments
import json  # serialize output as JSONL
from pathlib import Path  # handle filesystem paths
from typing import Callable, Dict, Iterable  # type annotations for clarity

from datasets import load_dataset  # fetch datasets from Hugging Face Hub


def format_gsm8k(example: Dict[str, str]) -> Dict[str, str]:  # format GSM8K sample into prompt/response
  question = example["question"].strip()  # pull question text and trim whitespace
  answer = example["answer"].strip()  # pull worked solution with final answer marker
  prompt = f"Solve the math problem step by step:\n{question}"  # craft instruction-style prompt
  response = f"Let's reason step by step.\n{answer}"  # preserve reasoning and final answer
  return {"prompt": prompt, "response": response}  # emit normalized record


def format_math_qa(example: Dict[str, str]) -> Dict[str, str]:  # format MathQA entries
  question = example.get("Problem", "").strip()  # question stem
  rationale = example.get("Rationale", "").strip()  # reasoning explanation
  correct = example.get("Correct Answer", "").strip()  # final choice letter
  options = example.get("options", "").strip()  # answer options string
  prompt_lines = ["Solve the multiple-choice math problem:", question]  # build prompt lines
  if options:  # include options when available
    prompt_lines.append(options)  # append option string
  prompt = "\n".join(prompt_lines)  # join prompt lines
  response_lines = ["Reasoning:", rationale, f"Final answer: {correct}"]  # build response with reasoning + answer
  response = "\n".join(line for line in response_lines if line.strip())  # join non-empty lines
  return {"prompt": prompt, "response": response}  # return formatted example


def format_mbpp(example: Dict[str, object]) -> Dict[str, str]:  # format MBPP programming tasks
  prompt = example.get("prompt", "").strip()  # natural language task description
  code = example.get("code", "").strip()  # reference solution code
  tests = example.get("test_list", [])  # list of unit tests
  test_block = "\n".join(tests) if tests else ""  # combine tests if present
  prompt_text = f"Write a Python function.\nTask:\n{prompt}"  # compose instruction prompt
  response_lines = ["Solution:", code]  # start response with solution code
  if test_block:  # include tests if available
    response_lines.append("Tests:")  # add header
    response_lines.append(test_block)  # add tests
  response = "\n".join(response_lines)  # join response lines
  return {"prompt": prompt_text, "response": response}  # emit normalized record


def format_science_qa(example: Dict[str, object]) -> Dict[str, str]:  # format ScienceQA style entries
  question = example.get("question", "").strip()  # question text
  choices = example.get("choices", [])  # answer choices list
  rationale = example.get("rationale", "").strip()  # explanation text
  answer_idx = example.get("answer", 0)  # index of correct choice
  choice_lines = [f"{idx}. {choice}" for idx, choice in enumerate(choices, start=1)]  # enumerate choices
  prompt_parts = ["Answer the science question with a rationale:", question]  # prompt header + question
  if choice_lines:  # include choices when present
    prompt_parts.append("Choices:")  # header
    prompt_parts.extend(choice_lines)  # append choice lines
  prompt = "\n".join(prompt_parts)  # join prompt
  if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):  # guard index
    final_answer = choices[answer_idx]  # map index to text
  else:
    final_answer = str(answer_idx)  # fallback to raw value
  response_lines = ["Rationale:", rationale, f"Final answer: {final_answer}"]  # response composition
  response = "\n".join(line for line in response_lines if line.strip())  # join non-empty lines
  return {"prompt": prompt, "response": response}  # return normalized example


FORMATTERS: Dict[str, Callable[[Dict[str, object]], Dict[str, str]]] = {  # map dataset id to formatter
  "gsm8k": format_gsm8k,  # math word problems with chain-of-thought
  "math_qa": format_math_qa,  # multiple-choice math reasoning
  "mbpp": format_mbpp,  # programming-by-example tasks
  "science_qa": format_science_qa,  # science questions with rationales
}  # close mapping

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {  # describe dataset source + config
  "gsm8k": {"path": "gsm8k", "config": "main"},  # GSM8K main split
  "math_qa": {"path": "math_qa", "config": None},  # MathQA default config
  "mbpp": {"path": "mbpp", "config": None},  # MBPP default config
  "science_qa": {"path": "derek-thomas/ScienceQA", "config": None},  # community ScienceQA fork
}


def build_dataset(name: str, split: str, output: Path, limit: int) -> None:  # orchestrate dataset export
  formatter = FORMATTERS[name]  # resolve formatter function
  registry = DATASET_REGISTRY[name]  # lookup dataset metadata
  path = registry["path"]  # dataset path on hub
  config = registry.get("config")  # optional config name
  if config:
    ds = load_dataset(path, config, split=split)  # load with config when specified
  else:
    ds = load_dataset(path, split=split)  # otherwise use default config
  output.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
  with output.open("w", encoding="utf-8") as handle:  # open target JSONL file
    for idx, example in enumerate(ds):  # iterate records
      if limit and idx >= limit:  # honor optional limit
        break  # stop when limit reached
      record = formatter(example)  # format example
      if not record.get("prompt") or not record.get("response"):  # skip empty entries
        continue  # skip invalid record
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")  # write JSON line


def parse_args() -> argparse.Namespace:  # build CLI parser
  parser = argparse.ArgumentParser(description="Generate reasoning JSONL from Hugging Face datasets")  # seed parser
  parser.add_argument("--dataset", choices=sorted(FORMATTERS.keys()), required=True, help="Dataset key to export")  # dataset choice
  parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")  # split option
  parser.add_argument("--output", type=Path, required=True, help="Path to write JSONL file")  # output path
  parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of examples")  # limit option
  return parser.parse_args()  # return parsed args


def main() -> None:  # entrypoint
  args = parse_args()  # parse CLI args
  build_dataset(args.dataset, args.split, args.output, args.limit)  # generate dataset


if __name__ == "__main__":  # script guard
  main()  # invoke main
