from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TOFUSample:
    text: str
    label: int  # 1=forget, 0=retain
    question: str
    answer: str
    split: str


class TOFUDataset:
    def __init__(self, forget_split: str = "forget01", retain_split: str = "retain99"):
        self.forget_split = forget_split
        self.retain_split = retain_split
        self._forget_data = None
        self._retain_data = None

    def _load(self):
        from datasets import load_dataset

        if self._forget_data is None:
            print(f"Loading TOFU forget split: {self.forget_split} ...")
            self._forget_data = load_dataset("locuslab/TOFU", self.forget_split)["train"]
            print(f"  Loaded {len(self._forget_data)} forget samples.")

        if self._retain_data is None:
            print(f"Loading TOFU retain split: {self.retain_split} ...")
            self._retain_data = load_dataset("locuslab/TOFU", self.retain_split)["train"]
            print(f"  Loaded {len(self._retain_data)} retain samples.")

    @staticmethod
    def _format(question: str, answer: str) -> str:
        return f"Question: {question} Answer: {answer}"

    def get_samples(
        self,
        n_forget: Optional[int] = None,
        n_retain: Optional[int] = None,
        seed: int = 42,
    ) -> List[TOFUSample]:
        self._load()

        forget_data = self._forget_data
        retain_data = self._retain_data

        if n_forget is not None and n_forget < len(forget_data):
            forget_data = forget_data.shuffle(seed=seed).select(range(n_forget))
        if n_retain is not None and n_retain < len(retain_data):
            retain_data = retain_data.shuffle(seed=seed).select(range(n_retain))

        samples: List[TOFUSample] = []
        for row in forget_data:
            q, a = row["question"], row["answer"]
            samples.append(
                TOFUSample(
                    text=self._format(q, a),
                    label=1,
                    question=q,
                    answer=a,
                    split=self.forget_split,
                )
            )
        for row in retain_data:
            q, a = row["question"], row["answer"]
            samples.append(
                TOFUSample(
                    text=self._format(q, a),
                    label=0,
                    question=q,
                    answer=a,
                    split=self.retain_split,
                )
            )
        return samples

    def get_forget_texts(self) -> List[str]:
        self._load()
        return [self._format(row["question"], row["answer"]) for row in self._forget_data]

    def get_retain_texts(self) -> List[str]:
        self._load()
        return [self._format(row["question"], row["answer"]) for row in self._retain_data]
