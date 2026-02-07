# %%
"""
python train_model.py --data-dir dev_phase/input_data/
"""

import time
from pathlib import Path
import torch

from tokam2d.metrics import compute_ap
from tokam2d.model import train_model
from tokam2d.tokam2d_utils import TokamDataset, vizualise_annotation

EVAL_SETS = ["train"]


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


# %%
def evaluate_model(model, data_dir):

    eval_dataset = TokamDataset(data_dir)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=2, collate_fn=collate_fn
    )

    model.eval()
    res = []
    for X, y in eval_dataloader:

        with torch.no_grad():
            y_pred = model(X)

        img = X[0][0].numpy()
        vizualise_annotation(img, y[0]["boxes"], y_pred[0]["boxes"])

        print(float(compute_ap(y_pred, y, threshold=0.5)))

    return res


def eval_model(data_dir):

    training_dir = data_dir / "train"
    print(training_dir)
    print("Training the model")
    start = time.time()
    model = train_model(training_dir)
    train_time = time.time() - start
    print(train_time)
    print("-" * 10)
    print("Evaluate the model")
    start = time.time()
    for eval_set in EVAL_SETS:
        _ = evaluate_model(model, data_dir / eval_set)
    test_time = time.time() - start
    print(test_time)


# %%

data_dir = "dev_phase/input_data"
eval_model(Path(data_dir))

# %%
