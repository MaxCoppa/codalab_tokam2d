# %%
import zipfile
from datetime import datetime
import os
import sys

file = "/home/onyxia/work/codalab_tokam2d/tokam2d/model/submission.py"

# Check the file exists
if not os.path.exists(file):
    print("submission.py not found")
    sys.exit(1)

# Create zip name with current date and hour
zip_file = datetime.now().strftime("submission.py.zip")

with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as z:
    # arcname ensures only "submission.py" appears inside the zip
    z.write(file, arcname="submission.py")

print(f"Compressed in: {zip_file}")

# %%
