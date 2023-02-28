#!/usr/bin/env python3

#region Imports

    #region Import Notice

import os, sys
ROOT = os.path.dirname(__file__)
depth = 1
for _ in range(depth): ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

    #endregion

from datasets import Dataset

from utils.preprocess import squad_format, from_squad

#endregion


#region Functional

def compose_datasets(dir, preprocess_train=True, preprocess_test=True, mode='train'):

    # we have two cases here: or we read and preprocess provided datasets, 
    # or we use those preprocessed in advance

    train_path = os.path.join(dir, 'train.jsonl')
    dev_path = os.path.join(dir, 'validation.jsonl')
    if preprocess_train:

        if os.path.exists(train_path):

            X_train = Dataset.from_pandas(
                squad_format(train_path)
            )
            # could leave only 'input_ids', 'attention_mask', 'start_positions', 'end_positions'
            # but will not remove columns as the unused ones are ignored
            # and the 'context', but will help us in postprocessing
            X_train = X_train.map(from_squad, batched=True)#, remove_columns=X.column_names)

        else: X_train = None
            
        if os.path.exists(dev_path):

            X_dev = Dataset.from_pandas(
                squad_format(dev_path)
            )
            X_dev = X_dev.map(from_squad, batched=True)

        else: X_dev = None

    else:
        X_train = Dataset.from_json(train_path) if os.path.exists(train_path) else None
        X_dev = Dataset.from_json(dev_path) if os.path.exists(dev_path) else None


    test_path = os.path.join(dir, 'input.jsonl')
    if preprocess_test:

        if os.path.exists(test_path):

            X_test = Dataset.from_pandas(
                squad_format(test_path, mode=mode)
            )
            X_test = X_test.map(from_squad, batched=True, fn_kwargs={'mode': mode})

        else: X_test = None

    else: X_test = Dataset.from_json(test_path) if os.path.exists(test_path) else None
        
    return X_train, X_dev, X_test

#endregion