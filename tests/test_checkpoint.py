"""An interrupted fit resumed from its checkpoints equals an uninterrupted one."""

import os

import numpy as np
import pytest

from boolmf import BayesianBooleanMF
from boolmf._sampler import engines

PARAMS = dict(n_chains=2, n_jobs=1, burn_in=30, n_draws=20, thin=1, max_sweeps=50,
              random_state=0, checkpoint_every=10)


class Interrupt(Exception):
    pass


@pytest.mark.parametrize("extra", [{}, dict(detection_effects=("sample",),
                                            background_effects=("sample",))])
def test_resumed_fit_equals_uninterrupted_fit(small_data, tmp_path, monkeypatch, extra):
    X, _ = small_data
    clean = BayesianBooleanMF(checkpoint_dir=tmp_path / "clean", **PARAMS, **extra).fit(X)
    assert os.listdir(tmp_path / "clean") == []          # finished chains leave nothing behind

    calls = {"n": 0}
    original = engines.CountEngine.update_rates

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 73:                             # second chain, sweep 23
            raise Interrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(engines.CountEngine, "update_rates", flaky)
    with pytest.raises(Interrupt):
        BayesianBooleanMF(checkpoint_dir=tmp_path / "run", **PARAMS, **extra).fit(X)
    assert len(os.listdir(tmp_path / "run")) == 1        # the interrupted chain's checkpoint
    monkeypatch.setattr(engines.CountEngine, "update_rates", original)

    resumed = BayesianBooleanMF(checkpoint_dir=tmp_path / "run", **PARAMS, **extra).fit(X)
    np.testing.assert_array_equal(resumed.log_likelihood_trace_, clean.log_likelihood_trace_)
    np.testing.assert_array_equal(resumed.components_, clean.components_)
    np.testing.assert_array_equal(resumed.activations_, clean.activations_)
    assert os.listdir(tmp_path / "run") == []


def test_checkpoint_from_other_settings_is_refused(small_data, tmp_path, monkeypatch):
    X, _ = small_data

    def stop(self, *args, **kwargs):
        raise Interrupt

    monkeypatch.setattr(engines.CountEngine, "update_rates", stop)
    params = {**PARAMS, "n_chains": 1, "checkpoint_every": 1}
    with pytest.raises(Interrupt):
        BayesianBooleanMF(checkpoint_dir=tmp_path, **params).fit(X)
    monkeypatch.undo()
    # no checkpoint is written before the first completed sweep
    assert os.listdir(tmp_path) == []
    calls = {"n": 0}
    original = engines.CountEngine.update_rates

    def later(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 5:
            raise Interrupt
        return original(self, *args, **kwargs)

    monkeypatch.setattr(engines.CountEngine, "update_rates", later)
    with pytest.raises(Interrupt):
        BayesianBooleanMF(checkpoint_dir=tmp_path, **params).fit(X)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="different data or settings"):
        BayesianBooleanMF(checkpoint_dir=tmp_path, **{**params, "likelihood": "noisy_or"}).fit(X)
    with pytest.raises(ValueError, match="int random_state"):
        BayesianBooleanMF(checkpoint_dir=tmp_path, random_state=None).fit(X)
