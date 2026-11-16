import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import kstest
from sklearn.exceptions import NotFittedError

from uniformizer import AutoLogitGMMUniformizer


def test_transform_approximates_uniform_for_a_well_fitted_distribution():
    rng = np.random.default_rng(42)
    logits = np.concatenate(
        (rng.normal(-2.0, 0.5, 3000), rng.normal(1.0, 0.8, 3000))
    )
    rng.shuffle(logits)
    probabilities = expit(logits)

    uniformizer = AutoLogitGMMUniformizer(max_components=3).fit(
        probabilities[:4000]
    )
    transformed = uniformizer.transform(probabilities[4000:])
    statistic = kstest(transformed, "uniform").statistic

    assert statistic < 0.04


def test_fit_transform_accepts_column_vectors_and_target():
    probabilities = np.linspace(0.0, 1.0, 64).reshape(-1, 1)
    target = np.zeros(probabilities.shape[0])

    uniformizer = AutoLogitGMMUniformizer(max_components=3)
    transformed = uniformizer.fit_transform(probabilities, target)

    assert transformed.shape == (probabilities.shape[0],)
    assert np.all(np.isfinite(transformed))
    assert np.all((transformed >= 0.0) & (transformed <= 1.0))
    assert 1 <= uniformizer.best_gmm_.n_components <= 3


def test_transform_requires_a_fitted_estimator():
    uniformizer = AutoLogitGMMUniformizer()

    with pytest.raises(NotFittedError):
        uniformizer.transform([0.2, 0.8])


def test_fit_requires_at_least_two_probabilities():
    with pytest.raises(ValueError, match="At least two probabilities"):
        AutoLogitGMMUniformizer().fit([0.2])


@pytest.mark.parametrize(
    "probabilities",
    [
        [],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.1, np.nan],
        [0.1, np.inf],
        [[0.1, 0.9]],
    ],
)
def test_fit_rejects_invalid_probability_inputs(probabilities):
    with pytest.raises(ValueError):
        AutoLogitGMMUniformizer().fit(probabilities)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("max_components", 0),
        ("max_components", 1.5),
        ("max_components", True),
        ("eps", 0.0),
        ("eps", 0.5),
        ("eps", np.nan),
    ],
)
def test_fit_rejects_invalid_configuration(parameter, value):
    uniformizer = AutoLogitGMMUniformizer(**{parameter: value})

    with pytest.raises(ValueError):
        uniformizer.fit([0.2, 0.8])


def test_component_search_does_not_exceed_distinct_observations():
    probabilities = [0.1, 0.1, 0.9, 0.9]
    uniformizer = AutoLogitGMMUniformizer(max_components=8).fit(probabilities)

    assert uniformizer.best_gmm_.n_components <= 2
