

import gzip

from tqdm import tqdm

import prior

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    # for split, size in zip(("train","val","test"),(100_000, 1_000, 1_000)):
    #     with gzip.open(f"dataset_files/procthor100k_balanced_partitionedassets_{split}.gzip", "r") as f:
    
    for split, size in zip(("train","val","test"),(20, 20, 20)):
        with gzip.open(f"{split}_tiny.gzip", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-100k", split=split
        )
    return prior.DatasetDict(**data)
