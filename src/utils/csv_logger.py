from pathlib import Path 
import pandas as pd

def save_csv(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )