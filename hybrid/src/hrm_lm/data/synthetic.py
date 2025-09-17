"""Synthetic arithmetic dataset for quick HRM-LM smoke tests."""  # module summary

import random  # random sampling for dataset
from typing import List, Tuple  # type hints for clarity

import torch  # tensor creation for batches

from .simple_tokenizer import SimpleTokenizer, PAD_ID, BOS_ID, EOS_ID  # reuse tokenizer and ids

Example = Tuple[List[int], List[int], List[int]]  # type alias for dataset entry

def build_synthetic_dataset(n: int = 1000, seed: int = 1337) -> Tuple[SimpleTokenizer, List[Example]]:
  random.seed(seed)  # seed rng for determinism
  pairs = []  # hold question-answer text pairs
  for _ in range(n):  # generate examples
    a = random.randint(10, 99)  # first operand
    b = random.randint(10, 99)  # second operand
    question = f"What is {a} + {b}?"  # build question string
    answer = str(a + b)  # compute answer
    pairs.append((question, answer))  # store pair
  tokenizer = SimpleTokenizer()  # instantiate tokenizer
  tokenizer.fit([q for q, _ in pairs] + [ans for _, ans in pairs])  # build vocab
  data: List[Example] = []  # encoded dataset
  for question, answer in pairs:  # encode each pair
    enc = tokenizer.encode(question, add_specials=True)  # encode question
    dec = tokenizer.encode(answer, add_specials=True)  # encode answer
    decoder_in = dec[:-1]  # decoder input includes BOS and drops EOS
    labels = dec[1:]  # labels start after BOS and keep EOS
    data.append((enc, decoder_in, labels))  # store example
  return tokenizer, data  # return tokenizer and dataset

def pad_batch(batch: List[Example], pad_id: int = PAD_ID) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  encs, dec_ins, labels = zip(*batch)  # unpack batch
  enc_len = max(len(seq) for seq in encs)  # max encoder length
  dec_len = max(len(seq) for seq in dec_ins)  # max decoder input length
  lab_len = max(len(seq) for seq in labels)  # max label length
  enc_tensor = torch.full((len(batch), enc_len), pad_id, dtype=torch.long)  # padded encoder tensor
  dec_in_tensor = torch.full((len(batch), dec_len), pad_id, dtype=torch.long)  # padded decoder input tensor
  label_tensor = torch.full((len(batch), lab_len), -100, dtype=torch.long)  # label tensor with ignore index
  for idx, (enc, dec_in, lab) in enumerate(batch):  # fill tensors
    enc_tensor[idx, :len(enc)] = torch.tensor(enc, dtype=torch.long)  # copy encoder ids
    dec_in_tensor[idx, :len(dec_in)] = torch.tensor(dec_in, dtype=torch.long)  # copy decoder ids
    label_tensor[idx, :len(lab)] = torch.tensor(lab, dtype=torch.long)  # copy labels
  enc_mask = (enc_tensor != pad_id).long()  # encoder attention mask
  dec_mask = (dec_in_tensor != pad_id).long()  # decoder attention mask
  return enc_tensor, dec_in_tensor, label_tensor, enc_mask, dec_mask  # return batch tensors
