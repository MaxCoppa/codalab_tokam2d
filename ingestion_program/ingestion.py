import json
import sys
import time
import pandas as pd
from pathlib import Path

import numpy as np

input_dir = Path('/app/input_data/')
output_dir = Path('/app/output/')
program_dir = Path('/app/program')
submission_dir = Path('/app/ingested_program')

EVAL_SETS = ["test", "private_test"]

sys.path.append(program_dir)
sys.path.append(submission_dir)


def evaluate_model(model, data_dir):
    eval_dataset = make_dataset(data_dir)
    eval_dataloader = torch.DataLoader(eval_dataset, batch_size=4)
    AP = 0

    model.eval()
    res = []
    for X, y in eval_dataset:
        y_pred = model(X)
        # Check how to make this work
        res.extend(y_pred)

    return res


def main():
    from submission import train_model

    training_dir = input_dir / "train"

    print("Training the model")
    start = time.time()
    model = train_model(training_dir)
    train_time = time.time() - start
    print('-' * 10)
    print('Evaluate the model')
    start = time.time()
    res = {}
    for eval_set in EVAL_SETS:
        res[eval_set] = evaluate_model(model, input_dir / eval_set)
    test_time = time.time() - start
    print('-' * 10)
    duration = train_time + test_time
    print(f'Completed Prediction. Total duration: {duration}')

    with open(output_dir / 'metadata.json', 'w+') as f:
        json.dump(dict(train_time=train_time, test_time=test_time), f)
    for eval_set in EVAL_SETS:
        with open(output_dir / f'{eval_set}_predictions.xml', 'w+') as f:
            ... # TODO: dump to XML res[eval_set]
    print()
    print('Ingestion Program finished. Moving on to scoring')


if __name__ == '__main__':
    main()
