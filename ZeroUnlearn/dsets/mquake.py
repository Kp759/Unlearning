import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"


class MQUAKEDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
    
        data_dir = Path(data_dir)
        saved_split_path = Path("data/mquake_data_saved_split.json")
        if saved_split_path.exists():
            with open(saved_split_path, "r") as f:
                self._data = json.load(f)
            if size is not None:
                self._data = self._data[:size]
        else:
            cf_loc = data_dir / ("MQuAKE-CF-3k-v2.json")

            with open(cf_loc, "r") as f:
                raw = json.load(f)
            data = []
            for i, record in enumerate(raw):
                data.append(
                    {
                        "case_id": i,
                        "requested_rewrite": record['requested_rewrite'],
                        "paraphrase_prompts": record["questions"],
                        "new_answer": record["new_answer"],
                        "answer": record["answer"],
                        "neighborhood_prompts": [],
                        "attribute_prompts": [],
                        "generation_prompts": [],
                    }
                )

            self._data = data[:size]
        # with open(data_dir / "mquake_data_saved.json", "w") as f:
        #     json.dump(self._data, f)
    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]
