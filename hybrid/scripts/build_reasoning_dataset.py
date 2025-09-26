#!/usr/bin/env python  # shebang for direct execution
"""Generate reasoning-focused prompts/responses from select Hugging Face datasets."""  # module docstring describing purpose

import argparse  # parse CLI arguments
import csv  # parse CSV-based datasets
import io  # treat downloaded payloads as streams
import json  # serialize output as JSONL
import zipfile  # extract archive datasets
from functools import lru_cache  # memoize remote downloads
from pathlib import Path  # handle filesystem paths
from typing import Callable, Dict, Iterable  # type annotations for clarity
from urllib.request import urlopen  # download remote resources

from datasets import load_dataset  # fetch datasets from Hugging Face Hub



COSMOS_BASE_URL = "https://github.com/wilburOne/cosmosqa/raw/master/data/"  # base URL for CosmosQA files
COSMOS_FILE_MAP = {  # map splits to remote filenames
  "train": "train.csv",  # training split CSV
  "validation": "valid.csv",  # validation split CSV
  "dev": "valid.csv",  # dev alias for validation
  "val": "valid.csv",  # additional alias
  "test": "test.jsonl",  # test split JSONL
}
MATHQA_ZIP_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"  # remote archive for MathQA


@lru_cache(maxsize=None)
def fetch_bytes(url: str) -> bytes:  # download remote resource with caching
  with urlopen(url) as handle:  # open URL stream
    return handle.read()  # return raw bytes


def iter_cosmos_examples(split: str):  # stream CosmosQA samples without HF dataset scripts
  key = split.lower()  # normalize split input
  filename = COSMOS_FILE_MAP.get(key, COSMOS_FILE_MAP.get(split, None))  # resolve filename with aliases
  if filename is None:  # guard against unsupported splits
    raise ValueError(f"Unsupported CosmosQA split '{split}'.")  # inform caller of invalid split
  data = fetch_bytes(f"{COSMOS_BASE_URL}{filename}")  # download targeted file
  if filename.endswith('.csv'):  # parse CSV formatted splits
    reader = csv.DictReader(io.StringIO(data.decode('utf-8')))  # decode bytes into CSV rows
    for row in reader:  # iterate rows
      yield {
        "id": row.get("id", ""),
        "context": row.get("context", ""),
        "question": row.get("question", ""),
        "answer0": row.get("answer0", ""),
        "answer1": row.get("answer1", ""),
        "answer2": row.get("answer2", ""),
        "answer3": row.get("answer3", ""),
        "label": int(row.get("label", -1)) if row.get("label") not in (None, "") else -1,
      }
  else:  # handle JSONL formatted test split
    for line in io.StringIO(data.decode('utf-8')):  # iterate lines lazily
      if not line.strip():
        continue
      sample = json.loads(line)  # parse JSON line
      if "label" in sample and isinstance(sample["label"], str):  # normalize label to int when provided
        try:
          sample["label"] = int(sample["label"])
        except ValueError:
          sample["label"] = -1
      yield sample


@lru_cache(maxsize=1)
def load_mathqa_archive() -> zipfile.ZipFile:  # cache MathQA archive in memory
  return zipfile.ZipFile(io.BytesIO(fetch_bytes(MATHQA_ZIP_URL)))  # download and open zip file


def iter_mathqa_examples(split: str):  # stream MathQA samples without HF dataset scripts
  normalized = split.lower()  # normalize input split
  name_map = {
    "train": "train.json",
    "validation": "dev.json",
    "dev": "dev.json",
    "val": "dev.json",
    "test": "test.json",
  }
  target = name_map.get(normalized)
  if target is None:
    raise ValueError(f"Unsupported MathQA split '{split}'.")
  archive = load_mathqa_archive()  # obtain cached zip archive
  with archive.open(target) as handle:  # open requested JSON file
    payload = json.load(handle)  # load JSON array
  for entry in payload:  # stream entries
    yield entry


def format_gsm8k(example: Dict[str, str]) -> Dict[str, str]:  # format GSM8K sample into prompt/response
  question = example["question"].strip()  # pull question text and trim whitespace
  answer = example["answer"].strip()  # pull worked solution with final answer marker
  prompt = f"Solve the math problem step by step:\n{question}"  # craft instruction-style prompt
  response = f"Let's reason step by step.\n{answer}"  # preserve reasoning and final answer
  return {"prompt": prompt, "response": response}  # emit normalized record


def format_math_qa(example: Dict[str, str]) -> Dict[str, str]:  # format MathQA entries
  question = example.get("Problem", "").strip()  # question stem
  rationale = example.get("Rationale", "").strip()  # reasoning explanation
  options = example.get("options", "").strip()  # answer options string
  correct = example.get("Correct Answer") or example.get("correct", "")  # final choice label
  correct = str(correct).strip()  # normalize answer label
  prompt_lines = ["Solve the multiple-choice math problem:", question]  # build prompt lines
  if options:  # include options when available
    prompt_lines.append(options)  # append option string
  prompt = "\n".join(prompt_lines)  # join prompt lines
  response_lines = ["Reasoning:", rationale]  # seed response with rationale
  if correct:  # append final answer when provided
    response_lines.append(f"Final answer: {correct}")
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


def format_ai2_arc(example: Dict[str, object]) -> Dict[str, str]:  # format AI2 ARC science QA entries
  question_field = example.get("question")  # question may be dict or plain string
  if isinstance(question_field, dict):  # handle legacy schema with nested stem/choices
    stem = str(question_field.get("stem", "")).strip()  # normalize question stem
    choices_block = question_field.get("choices", {})  # pull nested choices mapping
  else:
    stem = str(question_field or "").strip()  # treat question as plain string
    choices_block = example.get("choices", {})  # fall back to top-level choices mapping
  labels = [str(label).strip() for label in choices_block.get("label", [])]  # normalize choice labels
  texts = [str(text_value).strip() for text_value in choices_block.get("text", [])]  # normalize choice texts
  if not labels or not texts:  # fall back when nested layout missing
    raw_choices = example.get("choices", {})  # examine alternate layout
    if isinstance(raw_choices, dict):  # ensure mapping
      labels = [str(label).strip() for label in raw_choices.get("label", [])]  # normalize labels
      texts = [str(text_value).strip() for text_value in raw_choices.get("text", [])]  # normalize texts
  prompt_lines = ["Answer the science question using reasoning:"]  # seed prompt header
  context = example.get("context")  # optional supporting context
  if isinstance(context, str) and context.strip():  # include context when present
    prompt_lines.append(f"Context:\n{context.strip()}")  # append context block
  if stem:  # ensure question stem captured
    prompt_lines.append(f"Question:\n{stem}")  # append question text
  prompt_lines.append("Choices:")  # introduce answer options
  for label, text_value in zip(labels, texts):  # iterate label/text pairs
    if text_value:  # ignore empty options
      prompt_lines.append(f"{label}. {text_value}")  # append formatted choice
  prompt = "\n".join(line for line in prompt_lines if line.strip())  # join non-empty prompt lines
  answer_key = str(example.get("answerKey", "")).strip()  # fetch labeled correct answer
  lookup = {label: text_value for label, text_value in zip(labels, texts)}  # map label to text
  answer_text = lookup.get(answer_key, "")  # resolve answer text when available
  if answer_text:  # include supporting text when available
    response = f"Final answer: {answer_key}. {answer_text}"  # include label and text
  else:
    response = f"Final answer: {answer_key}"  # fall back to label only
  return {"prompt": prompt, "response": response}  # emit normalized record


def format_commonsense_qa(example: Dict[str, object]) -> Dict[str, str]:  # format CommonsenseQA samples
  question = example.get("question", "").strip()  # extract question text
  choices = example.get("choices", {})  # retrieve choices bundle
  labels = [label.strip() for label in choices.get("label", [])]  # normalize choice labels
  texts = [text_value.strip() for text_value in choices.get("text", [])]  # normalize choice texts
  prompt_lines = ["Answer the commonsense question:", question, "Choices:"]  # compose prompt header
  for label, text_value in zip(labels, texts):  # walk paired labels/texts
    prompt_lines.append(f"{label}. {text_value}")  # append choice description
  prompt = "\n".join(line for line in prompt_lines if line.strip())  # join prompt lines
  answer_key = example.get("answerKey", "").strip()  # load correct choice label
  lookup = {label: text_value for label, text_value in zip(labels, texts)}  # map label to text
  answer_text = lookup.get(answer_key, "")  # lookup answer text when present
  if answer_text:  # ensure we have supporting text
    response = f"Final answer: {answer_key}. {answer_text}"  # record label and text
  else:
    response = f"Final answer: {answer_key}"  # fallback to label only
  return {"prompt": prompt, "response": response}  # return normalized record


def format_cosmos_qa(example: Dict[str, object]) -> Dict[str, str]:  # format CosmosQA narrative reasoning samples
  context = example.get("context", "").strip()  # fetch narrative context
  question = example.get("question", "").strip()  # fetch question text
  answers = [example.get(f"answer{i}", "").strip() for i in range(4)]  # collect choice texts
  prompt_lines = ["Read the passage and answer the question:"]  # seed prompt description
  if context:  # include context when present
    prompt_lines.append(f"Passage:\n{context}")  # append passage
  if question:  # include question
    prompt_lines.append(f"Question:\n{question}")  # append question
  prompt_lines.append("Choices:")  # add choices header
  labels = [chr(ord('A') + idx) for idx in range(len(answers))]  # derive letter labels
  for label, text_value in zip(labels, answers):  # iterate label/text pairs
    if text_value:  # skip empty options
      prompt_lines.append(f"{label}. {text_value}")  # append populated choice
  prompt = "\n".join(line for line in prompt_lines if line.strip())  # build prompt string
  label_index = example.get("label")  # grab integer answer index
  if isinstance(label_index, int) and 0 <= label_index < len(answers):  # validate label index
    answer_label = labels[label_index]  # derive answer label
    answer_text = answers[label_index]  # derive answer text
  else:
    answer_label = ""  # fallback when unavailable
    answer_text = ""  # fallback when unavailable
  if answer_label and answer_text:  # standard case with label/text
    response = f"Final answer: {answer_label}. {answer_text}"  # include label and text
  elif answer_label:  # fallback when only label exists
    response = f"Final answer: {answer_label}"  # record label only
  else:
    response = "Final answer: "  # final fallback when missing data
  return {"prompt": prompt, "response": response}  # return normalized record


def format_alpaca_cleaned(example: Dict[str, str]) -> Dict[str, str]:  # format Alpaca instruction-following data
  instruction = example.get("instruction", "").strip()  # extract instruction text
  input_text = example.get("input", "").strip()  # optional input field
  prompt_lines = ["Instruction:", instruction]  # seed prompt with instruction header
  if input_text:  # include input when provided
    prompt_lines.extend(["Input:", input_text])  # append input section
  prompt_lines.append("Response:")  # signal where the model should respond
  prompt = "\n".join(line for line in prompt_lines if line.strip())  # combine non-empty segments
  response = example.get("output", "").strip()  # target assistant output
  return {"prompt": prompt, "response": response}  # emit normalized record


def format_dolly(example: Dict[str, str]) -> Dict[str, str]:  # format Databricks Dolly examples
  instruction = example.get("instruction", "").strip()  # instruction text
  context = example.get("context", "").strip()  # optional background context
  prompt_lines = ["Instruction:", instruction]  # begin prompt with instruction
  if context:  # append context when present
    prompt_lines.extend(["Context:", context])  # add context section
  prompt_lines.append("Response:")  # add response cue
  prompt = "\n".join(line for line in prompt_lines if line.strip())  # join prompt lines
  response = example.get("response", "").strip()  # assistant response
  return {"prompt": prompt, "response": response}  # return normalized example


def format_hh_rlhf(example: Dict[str, str]) -> Dict[str, str]:  # format Anthropic HH-RLHF conversations
  chosen = example.get("chosen", "")  # retrieve preferred conversation transcript
  marker = "Assistant:"  # define assistant marker
  last_idx = chosen.rfind(marker)  # locate final assistant segment
  if last_idx == -1:  # ensure marker present
    return {"prompt": "", "response": ""}  # signal invalid record for upstream skip
  prompt = chosen[: last_idx + len(marker)].strip()  # keep conversation up to assistant cue
  response = chosen[last_idx + len(marker):].strip()  # capture assistant reply text
  return {"prompt": prompt, "response": response}  # emit prompt/response pair

def format_openhermes(example: Dict[str, object]) -> Dict[str, str]:  # format OpenHermes multi-turn chats
  convo = example.get("conversations") or []  # retrieve conversation turns
  if not convo:  # guard empty conversations
    return {"prompt": "", "response": ""}
  role_titles = {"human": "Human", "gpt": "Assistant", "system": "System"}  # role labeling map
  cleaned = [(role_titles.get(turn.get("from"), "Unknown"), str(turn.get("value", "")).strip()) for turn in convo]  # normalize
  cleaned = [(role, value) for role, value in cleaned if value]  # drop blank entries
  if len(cleaned) < 2 or cleaned[-1][0] != "Assistant":  # ensure final assistant reply exists
    return {"prompt": "", "response": ""}
  prompt_lines = [f"{role}: {value}" for role, value in cleaned[:-1]]  # all turns except last as prompt
  prompt = "\n\n".join(prompt_lines)  # double newline separates turns
  response = cleaned[-1][1]  # final assistant answer
  return {"prompt": prompt, "response": response}  # return formatted dialog

FORMATTERS: Dict[str, Callable[[Dict[str, object]], Dict[str, str]]] = {  # map dataset id to formatter
  "ai2_arc": format_ai2_arc,  # science multiple-choice with commonsense requirements
  "alpaca_cleaned": format_alpaca_cleaned,  # instruction-following English prose
  "commonsense_qa": format_commonsense_qa,  # general commonsense reasoning
  "cosmos_qa": format_cosmos_qa,  # narrative-based commonsense reasoning
  "dolly": format_dolly,  # Dolly instruction dataset
  "gsm8k": format_gsm8k,  # math word problems with chain-of-thought
  "hh_rlhf": format_hh_rlhf,  # helpful/harmless conversational data
  "math_qa": format_math_qa,  # multiple-choice math reasoning
  "openhermes": format_openhermes,  # high-quality conversational English
  "mbpp": format_mbpp,  # programming-by-example tasks
}  # close mapping

DATASET_REGISTRY: Dict[str, Dict[str, object]] = {  # describe dataset source + config
  "ai2_arc": {"path": "allenai/ai2_arc", "config": "ARC-Challenge"},  # ARC challenge split
  "alpaca_cleaned": {"path": "yahma/alpaca-cleaned", "config": None},  # Alpaca instruction dataset
  "commonsense_qa": {"path": "tau/commonsense_qa", "config": None},  # CommonsenseQA default config
  "cosmos_qa": {"loader": "cosmos_remote"},  # CosmosQA custom loader
  "dolly": {"path": "databricks/databricks-dolly-15k", "config": None},  # Dolly instruction dataset
  "gsm8k": {"path": "gsm8k", "config": "main"},  # GSM8K main split
  "hh_rlhf": {"path": "Anthropic/hh-rlhf", "config": None},  # Anthropic HH-RLHF conversations
  "math_qa": {"loader": "mathqa_remote"},  # MathQA custom loader
  "mbpp": {"path": "mbpp", "config": None},  # MBPP default config
  "openhermes": {"path": "teknium/OpenHermes-2.5", "config": None},  # OpenHermes instruction dataset
}


def build_dataset(name: str, split: str, output: Path, limit: int) -> None:  # orchestrate dataset export
  formatter = FORMATTERS[name]  # resolve formatter function
  registry = DATASET_REGISTRY[name]  # lookup dataset metadata
  loader = registry.get("loader", "hf" )  # determine loader strategy
  if loader == "cosmos_remote":  # custom CosmosQA loader
    ds = iter_cosmos_examples(split)  # stream CosmosQA rows
  elif loader == "mathqa_remote":  # custom MathQA loader
    ds = iter_mathqa_examples(split)  # stream MathQA rows
  else:
    path = registry["path"]  # dataset path on hub
    config = registry.get("config")  # optional config name
    load_kwargs = {"split": split}  # base load arguments
    if registry.get("trust_remote_code"):  # enable remote code when required
      load_kwargs["trust_remote_code"] = True  # opt-in to remote dataset scripts
    if config:  # config present
      ds = load_dataset(path, config, **load_kwargs)  # load with config when specified
    else:
      ds = load_dataset(path, **load_kwargs)  # otherwise use default config
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
