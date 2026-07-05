"""
Three standard anomaly detectors from PyOD, standing in for the families a CPS
detector is usually built from:

  - Isolation Forest : global / tree-based statistical detector.
  - LOF              : local density; the method that *should* cope with
                       multimodality but degrades under strong imbalance.
  - AutoEncoder      : a single deep reconstruction model (no cluster structure)
                       - the direct ablation that over-interpolates between modes.

Each returns train scores and test scores (higher = more anomalous), matching
the interface used for the VaDE / sequential-VAE models.
"""

from __future__ import annotations

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF


def _fit_score(model, x_train, x_test):
    model.fit(x_train)
    return model.decision_function(x_train), model.decision_function(x_test)


def run_iforest(x_train, x_test, seed=0):
    return _fit_score(IForest(n_estimators=200, random_state=seed), x_train, x_test)


def run_lof(x_train, x_test, seed=0):
    # novelty=True so LOF scores unseen test points against the train manifold.
    return _fit_score(LOF(n_neighbors=30, novelty=True), x_train, x_test)


def run_autoencoder(x_train, x_test, seed=0):
    """PyOD's deep AutoEncoder. The constructor signature changed across PyOD
    versions, so try the current one and fall back."""
    from pyod.models.auto_encoder import AutoEncoder
    try:
        model = AutoEncoder(hidden_neuron_list=[128, 64, 64, 128],
                            epoch_num=40, batch_size=256, random_state=seed,
                            verbose=0)
    except TypeError:
        model = AutoEncoder(random_state=seed)  # older API / default arch
    return _fit_score(model, x_train, x_test)
