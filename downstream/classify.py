#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import os

from web_trainer import *

#endregion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#region Functional

def fine_tune(instruction, X_train=None, X_dev=None):
    label_mapping = instruction['label_mapping']
    id_mapping = {value: key for key, value in label_mapping.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        instruction['input_model_path'],
        num_labels=len(label_mapping),
        label2id=label_mapping,
        id2label=id_mapping
        ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(instruction['input_model_path'])
    training_args = WebClassificationTrainingArguments(**instruction['training_kwargs']).configured
    trainer = WebClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_dev,
        **instruction['trainer_kwargs']
    ).configured
    trainer.train()
    dir_name = os.path.dirname(instruction['output_model_path'])
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    trainer.save_model(dir_name)


def build_model(instruction, X_train=None, X_dev=None, mode='train'):
    if instruction['fine-tune'] and mode == 'train': fine_tune(instruction, X_train, X_dev)
    label_mapping = instruction['label_mapping']
    id_mapping = {value: key for key, value in label_mapping.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        instruction['output_model_path'],
        num_labels=len(label_mapping),
        label2id=label_mapping,
        id2label=id_mapping
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(instruction['output_model_path'])
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)
    return classifier


def classify(X_test, classifier):
    inputs = classifier.tokenizer(
        X_test['text'],
        truncation=True,
        padding='longest', # True
        add_special_tokens=True,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad(): output = classifier.model(**inputs)
    return {
        'logits': output['logits']
    } # datasets.Dataset.map() requires the function to return a dict

#endregion