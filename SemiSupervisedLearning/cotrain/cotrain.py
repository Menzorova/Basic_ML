from functools import partial
from typing import List, Optional

import numpy as np
import warnings

from .datasets import SSLTrainSet

def probs_to_margin(probs: np.ndarray, margin_mode: str = "soft") -> np.ndarray:
    probs = np.array(probs)
    if margin_mode == "hard":
        probs = np.argmax(probs, axis=-1)
        probs[probs == 0] = -1
        return np.abs(probs.mean(axis=0))
    elif margin_mode == "soft":
        if probs.shape[-1] != 2:
            warnings.warn("Cannot perform soft labelling. Recalculating in hard mode.")
            return probs_to_margin(probs, "hard")
        return np.abs(np.subtract(*np.mean(probs, axis=0).T))
    else:
        raise ValueError

class Optimum(object):
    """Storage for optimum solution.

    Attributes:
        point: Optimal point.
        score: Functional score.
    """

    def __init__(self) -> None:
        self.point = None  # Optional[float]
        self.score = np.inf


def infer_theta(margins: np.ndarray) -> float:
    """Calaculate theta w.r.t. SLA* algorithm.

    Arguments:
        margins: Margin (n_estimators x n_samples)

    Notes:
        https://hal.archives-ouvertes.fr/hal-01301617.
    """
    optimum_theta = Optimum()
    optimum_gamma = Optimum()

    gammas = np.linspace(1e-3, 1, 50)
    thetas = np.linspace(margins.min(), margins.max(), 50)

    # Optimize joint bayes risk.
    for theta in thetas:
        for gamma in gammas:
            # START: Calculate joint bayes risk, eq. (6).
            mask_theta_leq = margins <= theta
            mean_theta_leq = np.mean(margins * mask_theta_leq)

            mask_gamma_l = margins < gamma
            mean_gamma_l = np.mean(margins * mask_gamma_l)

            # The first term, which is Gibbs risk, is just an estimation.
            k_u = 0.5 + 0.5 * (np.mean(margins) - 1)
            k_u += mean_theta_leq - mean_gamma_l
            if k_u <= 0:
                k_u = 0
            else:
                k_u /= gamma

            gamma_score = k_u + np.mean(np.logical_and(mask_gamma_l, ~mask_theta_leq))
            # END: Calculate joint bayes risk, eq. (6).

            if optimum_gamma.score > gamma_score:
                optimum_gamma.score = gamma_score
                optimum_gamma.point = gamma

        prob = np.mean(margins > theta)
        if prob == 0:
            theta_score = np.inf
        else:
            theta_score = optimum_gamma.score / prob

        if optimum_theta.score >= theta_score:
            optimum_theta.score = theta_score
            optimum_theta.point = theta

    return optimum_theta.point


def get_utol_mask(
    model,
    udata: np.ndarray,
    co_mode: str = "sla",
    co_size: int = 10,
    margin_mode: str = "soft",
) -> np.ndarray:
    """Get mask from unlabelled data to labelled data."""
    if co_mode == "sla":
        margins = probs_to_margin(
            [e.predict_proba(udata) for e in model.estimators_], margin_mode
        )
        pos = margins >= infer_theta(margins)
    elif co_mode in ["best", "random"]:
        probs = model.predict(udata)
        if co_mode == "best":
            pos = np.argsort(np.max(probs, axis=0))[::-1][:co_size]
        else:
            if len(probs) >= co_size:
                pos = np.random.choice(len(probs), size=co_size, replace=False)
            else:
                pos = np.arange(len(probs))
    return pos


class BinaryCoTrainer(object):
    def __init__(
        self,
        models: List,
        co_mode: str = "sla",
        co_size: int = 10,
        margin_mode: str = "soft",
        max_iter: int = 100,
    ) -> None:
        """
        Arguments:
            models: Models to co-train.
            сo_mode: Option to choose number of samples to interchange.
                For `co_mode`: best, random.
            co_size: Number of samples to interchange.
            margin_mode: Option to calculate margin.
            max_iter: Maximum number of train iterations.
        """
        if len(models) != 2:
            raise ValueError(f"There should be 2 models, got {len(models)}.")
        self.models = models
        self.co_mode = co_mode
        self.co_size = co_size
        self.margin_mode = margin_mode
        self.max_iter = max_iter

        self.__get_mask = partial(
            get_utol_mask, co_mode=co_mode, co_size=co_size, margin_mode=margin_mode
        )

    def _fit(self, train_sets: List) -> None:
        masks = []
        new_ldatas = []
        for i in range(2):
            self.models[i].fit(train_sets[i].all_ldata, train_sets[i].all_labels)
            masks.append(self.__get_mask(self.models[i], train_sets[i].udata))
            new_ldatas.append(train_sets[i].utol(masks[i]))
            train_sets[i]._compress_udata(masks[i])

        for i, items in enumerate(zip(new_ldatas[::-1], self.models[::-1])):
            new_ldata, model = items
            train_sets[i]._extend_pseudo_ldata(new_ldata)
            train_sets[i]._extend_pseudo_labels(model.predict(new_ldata))

    def fit(self, train_sets: List) -> None:
        """
        Arguments:
            train_sets: Semi-supervised training set.
        """
        if len(train_sets) != 2:
            raise ValueError(f"There should be 2 train sets, got {len(train_sets)}.")

        for i in range(self.max_iter):
            if len(train_sets[0].udata) and len(train_sets[1].udata):
                self._fit(train_sets)
            else:
                break

    def predict(self, x_test: np.ndarray) -> List:
        preds = []
        for model in self.models:
            preds.append(model.predict(x_test))
        return preds
