from pathlib import Path 
import json
import pandas as pd
from torch.utils.data import Dataset

class CordDataset(Dataset):
    def __init__(
        self, 
        parquet_files,
        transform=None
    ):
        self.transform = transform
        dfs = [pd.read_parquet(file) for file in parquet_files]
        self.data = pd.concat(dfs, ignore_index=True)

    def __len__(
        self
    ):
        return len(self.data)

    def __getitem__(
        self,
        idx
    ):
        row = self.data.iloc[idx]
        image = row["image"]
        ground_truth = json.loads(row["ground_truth"])

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "ground_truth": ground_truth
        }
