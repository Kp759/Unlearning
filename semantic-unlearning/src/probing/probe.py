import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    layer_idx: int
    accuracy: float
    auc: float
    report: str
    n_train: int
    n_test: int
    coef: np.ndarray


class LinearProbe:
    def __init__(
        self,
        layer_idx: int,
        C: float = 1.0,
        max_iter: int = 1000,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.layer_idx = layer_idx
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.seed = seed
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[LogisticRegression] = None

    def fit_eval(self, X_forget: np.ndarray, X_retain: np.ndarray) -> ProbeResult:
        X = np.concatenate([X_forget, X_retain], axis=0)
        y = np.concatenate([
            np.ones(len(X_forget), dtype=int),
            np.zeros(len(X_retain), dtype=int),
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.seed,
        )
        self.clf.fit(X_train_s, y_train)

        y_pred = self.clf.predict(X_test_s)
        y_prob = self.clf.predict_proba(X_test_s)[:, 1]
        accuracy = float((y_pred == y_test).mean())
        auc = float(roc_auc_score(y_test, y_prob))
        report = classification_report(y_test, y_pred, target_names=["retain", "forget"])

        return ProbeResult(
            layer_idx=self.layer_idx,
            accuracy=accuracy,
            auc=auc,
            report=report,
            n_train=len(X_train),
            n_test=len(X_test),
            coef=self.clf.coef_[0].copy(),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None, "Call fit_eval first."
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.clf is not None, "Call fit_eval first."
        return self.clf.predict_proba(self.scaler.transform(X))

    def concept_direction(self) -> np.ndarray:
        assert self.clf is not None, "Call fit_eval first."
        w = self.clf.coef_[0]
        return w / (np.linalg.norm(w) + 1e-12)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "LinearProbe":
        with open(path, "rb") as f:
            return pickle.load(f)


class LayerwiseProber:
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.test_size = test_size
        self.seed = seed

    def run(
        self,
        forget_states: Dict[int, np.ndarray],
        retain_states: Dict[int, np.ndarray],
        save_dir: Optional[str] = None,
    ) -> Dict[int, ProbeResult]:
        results: Dict[int, ProbeResult] = {}
        layers = sorted(forget_states.keys())

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        for layer_idx in layers:
            probe = LinearProbe(
                layer_idx=layer_idx,
                C=self.C,
                max_iter=self.max_iter,
                test_size=self.test_size,
                seed=self.seed,
            )
            result = probe.fit_eval(forget_states[layer_idx], retain_states[layer_idx])
            results[layer_idx] = result
            print(f"  Layer {layer_idx:3d} | acc={result.accuracy:.4f} | auc={result.auc:.4f}")

            if save_dir is not None:
                probe.save(str(Path(save_dir) / f"probe_layer_{layer_idx:03d}.pkl"))

        return results

    @staticmethod
    def summary(results: Dict[int, ProbeResult], threshold: float = 0.7) -> dict:
        accuracies = {layer: r.accuracy for layer, r in results.items()}
        best_layer = max(accuracies, key=accuracies.__getitem__)
        return {
            "max_accuracy": accuracies[best_layer],
            "max_accuracy_layer": best_layer,
            "mean_accuracy": float(np.mean(list(accuracies.values()))),
            "layers_above_threshold": [l for l, a in accuracies.items() if a >= threshold],
        }
