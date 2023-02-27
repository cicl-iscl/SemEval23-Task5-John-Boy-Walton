#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from web_trainer import *

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#region Functional

def build_model(model_name):
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return model, tokenizer


def summarize(texts, summarizer, tokenizer):
    inputs = tokenizer(
        texts, 
        truncation=True,
        padding='longest', # True
        add_special_tokens=True,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad(): output = summarizer.generate(max_new_tokens=64, **inputs)
    summarized = tokenizer.batch_decode(output, skip_special_tokens=True)
    return {'summarized': summarized} # datasets.Dataset.map() requires the function to return a dict

#endregion