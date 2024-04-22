from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseMetric(ABC):
    """Base class for metrics."""

    @abstractmethod
    def _score(self, diff_true: List[str], diff_pred: List[str], full_dataset: pd.DataFrame):
        pass

    def score(
        self, diff_true: List[str], diff_pred: List[str], full_dataset: pd.DataFrame, n_samples: Optional[int] = None
    ):
        diff_true = list(diff_true)
        diff_pred = list(diff_pred)
        # If requested to use a sample
        if n_samples is not None:
            selected_indices = np.random.choice(len(diff_true), size=n_samples, replace=False)

            diff_true = [diff_true[i] for i in selected_indices]
            diff_pred = [diff_pred[i] for i in selected_indices]
            full_dataset = full_dataset.iloc[selected_indices]
        return self._score(diff_true, diff_pred, full_dataset)


class BaseSentenceMetric(BaseMetric):
    """Base class for metrics that score a single sentence at a time."""

    @abstractmethod
    def _score_single(self, diff_true: str, diff_pred: str, full_row: Dict):
        pass

    def _accum(self, objs: List[Any]):
        return sum(objs) / len(objs)

    def _score(self, diff_true: List[str], diff_pred: List[str], full_dataset: pd.DataFrame):
        scores = []
        inp = zip(diff_true, diff_pred, full_dataset.itertuples(index=False))
        for true, pred, full_row in tqdm(
            inp, desc=self.__class__.__name__, total=len(diff_true), leave=False, position=1
        ):
            # noinspection PyProtectedMember
            scores.append(self._score_single(true, pred, full_row._asdict()))
        return self._accum(scores)
