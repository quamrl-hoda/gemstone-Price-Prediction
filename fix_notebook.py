import json
import os

notebook_path = r"C:\Users\quamr\OneDrive\Desktop\project\gemstonePricePrediction\research\trials.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if this is the import cell
        is_import_cell = any("from gemstonePricePrediction.constants import *" in line for line in source)
        if is_import_cell:
            print("Found import cell, adding path fix...")
            # Prepend the path fix
            new_lines = [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "sys.path.append(str(Path(os.getcwd()) / 'src'))\n"
            ]
            # Avoid adding if already present (idempotency)
            if not any("sys.path.append" in line for line in source):
                cell['source'] = new_lines + source
                print("Added path fix.")
            else:
                print("Path fix already present.")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
