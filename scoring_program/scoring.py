import json
from pathlib import Path

from torchvision.metrics import compute_ap
from torchvision.datasets import load_xml

APP_DIR = Path("/app")
prediction_dir = APP_DIR / 'input' / 'res'
reference_dir = APP_DIR / 'input' / 'ref'

EVAL_SETS = ["test", "private_test"]

scores = {}
for eval_set in EVAL_SETS:
    print(f'Scoring {eval_set}')

    predictions = load_xml(prediction_dir / f'{eval_set}_prediction.xml'))
    targets = load_xml(reference_dir / f'{eval_set}_labels.xml')
    res[eval_set] = compute_ap(predictions, targets)

# Add train and test times in the score
with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    durations = json.load(f)
scores.update(**durations)
print(scores)

# Write output scores
(APP_DIR / 'output' / 'scores.json').write_text(json.dumps(scores))
