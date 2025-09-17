"""Simple whitespace tokenizer for synthetic arithmetic tasks."""  # describe module purpose

from typing import List  # import typing for type hints

PAD_ID = 0  # reserved pad token id
BOS_ID = 1  # reserved begin-of-sequence id
EOS_ID = 2  # reserved end-of-sequence id

class SimpleTokenizer:
  """Minimal tokenizer that assigns IDs per whitespace token."""  # class summary

  def __init__(self) -> None:
    self.pad_id = PAD_ID  # expose pad id attribute
    self.bos_id = BOS_ID  # expose bos id attribute
    self.eos_id = EOS_ID  # expose eos id attribute
    self.stoi = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}  # token to id mapping
    self.itos = {idx: tok for tok, idx in self.stoi.items()}  # id to token mapping

  def fit(self, texts: List[str]) -> None:
    for text in texts:  # iterate over input texts
      for token in text.strip().split():  # split whitespace tokens
        if token not in self.stoi:  # check for unseen token
          idx = len(self.stoi)  # assign next id
          self.stoi[token] = idx  # store mapping
          self.itos[idx] = token  # store reverse mapping

  def encode(self, text: str, add_specials: bool = True) -> List[int]:
    ids: List[int] = []  # initialize id list
    for token in text.strip().split():  # iterate tokens
      if token in self.stoi:  # known token branch
        ids.append(self.stoi[token])  # append existing id
      else:  # unseen token branch
        idx = len(self.stoi)  # compute new id
        self.stoi[token] = idx  # store mapping
        self.itos[idx] = token  # update reverse map
        ids.append(idx)  # append new id
    if add_specials:  # optionally add BOS/EOS
      ids = [self.bos_id] + ids + [self.eos_id]  # wrap sequence with specials
    return ids  # return encoded ids

  def decode(self, ids: List[int], skip_specials: bool = True) -> str:
    tokens: List[str] = []  # accumulate decoded tokens
    for idx in ids:  # iterate ids
      if skip_specials and idx in (self.pad_id, self.bos_id, self.eos_id):  # skip specials if flagged
        continue  # ignore this token
      tokens.append(self.itos.get(idx, "<unk>"))  # append token string
    return " ".join(tokens)  # join tokens back into text
