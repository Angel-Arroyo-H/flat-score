from numbers import Integral, Real
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logit
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted


class AutoLogitGMMUniformizer(BaseEstimator, TransformerMixin):
    """Uniformize one-dimensional probabilities using a logit-space GMM."""

    def __init__(
        self,
        max_components: int = 8,
        eps: float = 1e-12,
        random_state: Optional[int] = 42,
    ):
        self.max_components = max_components
        self.eps = eps
        self.random_state = random_state

    @staticmethod
    def _as_probability_vector(p: ArrayLike) -> np.ndarray:
        values = np.asarray(p, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise ValueError("Expected a 1D array or a 2D single-column array.")
        if values.size == 0:
            raise ValueError("Expected at least one probability.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Probabilities must all be finite.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Probabilities must be between 0 and 1.")
        return values

    def _to_logit(self, p: ArrayLike) -> np.ndarray:
        values = self._as_probability_vector(p)
        p_clipped = np.clip(values, self.eps, 1.0 - self.eps)
        return logit(p_clipped).reshape(-1, 1)

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "AutoLogitGMMUniformizer":
        _ = y
        if hasattr(self, "best_gmm_"):
            del self.best_gmm_
            del self.best_bic_

        if (
            isinstance(self.max_components, bool)
            or not isinstance(self.max_components, Integral)
            or self.max_components < 1
        ):
            raise ValueError("max_components must be a positive integer.")
        if (
            isinstance(self.eps, bool)
            or not isinstance(self.eps, Real)
            or not np.isfinite(self.eps)
            or not 0.0 < self.eps < 0.5
        ):
            raise ValueError("eps must be a finite number between 0 and 0.5.")

        z_train = self._to_logit(X)
        if z_train.shape[0] < 2:
            raise ValueError("At least two probabilities are required to fit.")
        max_components = min(self.max_components, np.unique(z_train).size)
        best_bic = np.inf
        best_gmm = None

        for k in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=self.random_state,
                n_init=3,
            )
            gmm.fit(z_train)
            bic = gmm.bic(z_train)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        if best_gmm is None:
            raise RuntimeError("Could not fit a Gaussian mixture to the data.")

        self.best_gmm_ = best_gmm
        self.best_bic_ = best_bic
        self.n_features_in_ = 1
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, attributes=["best_gmm_"])
        z = self._to_logit(X).ravel()
        weights = self.best_gmm_.weights_
        means = self.best_gmm_.means_.ravel()
        standard_deviations = np.sqrt(self.best_gmm_.covariances_.ravel())

        transformed = np.zeros_like(z, dtype=float)
        for weight, mean, standard_deviation in zip(
            weights, means, standard_deviations
        ):
            transformed += weight * norm.cdf(
                z, loc=mean, scale=standard_deviation
            )
        return transformed