import torch
from datasets import Dataset

from preprocess import *
from configure_model import *

if torch.cuda.is_available():
    cuda = torch.device('cuda')
    torch.cuda.set_device(cuda)


def main():

    TRAIN_PATH = 'webis22/train.jsonl'
    X = Dataset.from_pandas(
        squad_format(TRAIN_PATH)
    )
    # could leave only 'input_ids', 'attention_mask', 'start_positions', 'end_positions'
    # but will not remove columns as the unused ones are ignored
    # and the 'context', but will help us in post-processing
    X = X.map(from_squad, batched=True)#, remove_columns=X.column_names) 
    X_train, X_dev = split_dataset(X, test_size=0.075)
    
    # use defaults
    qa_trainer = CTrainer(
        train_dataset=X_train,
        eval_dataset=X_dev
    ).configured
    qa_trainer.train()
    qa_trainer.save_model('models/roberta-base-squad2-webis')

if __name__ == '__main__':
    main()
