import os
import zipfile
from pathlib import Path

import gdown

SCRIPT_DIR = Path(__file__).parent

URL = (
    "https://drive.google.com/file/d/1e0KubZJ6TTV74mHVUMHyrUwcbPlhyLVL/view?usp=sharing"
)
OUTPUT_FILENAME = str(SCRIPT_DIR / "baked_artifacts.zip")
EXTRACT_DIR = str(SCRIPT_DIR)

gdown.download(URL, OUTPUT_FILENAME, quiet=False, fuzzy=True)

with zipfile.ZipFile(OUTPUT_FILENAME, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

os.remove(OUTPUT_FILENAME)
