#!/usr/bin/env python3

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

from web_trainer import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(model_name):
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return model, tokenizer


def summarize(texts, summarizer, tokenizer):
    inputs = tokenizer(
            texts, 
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        ).to(DEVICE)
    output = summarizer.generate(max_new_tokens=64, **inputs)
    summarized = tokenizer.batch_decode(output, skip_special_tokens=True)
    return {'summarized': summarized} # datasets.Dataset.map() requires the function to return a dict