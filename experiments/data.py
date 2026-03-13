"""Dataset loaders for real-data experiments."""

import numpy as np
import pandas as pd

_BASE = "https://vincentarelbundock.github.io/Rdatasets/csv"


def load_faithful() -> tuple[np.ndarray, np.ndarray]:
    """Old Faithful geyser data (n=272).

    Returns (x, y) sorted ascending by x, where
        x = waiting time between eruptions (minutes)
        y = eruption duration (minutes)
    """
    df = pd.read_csv(f"{_BASE}/datasets/faithful.csv")
    x = df["waiting"].to_numpy(dtype=float)
    y = df["eruptions"].to_numpy(dtype=float)
    idx = np.argsort(x)
    return x[idx], y[idx]


def load_mcycle() -> tuple[np.ndarray, np.ndarray]:
    """Motorcycle accident head acceleration data (n=133).

    Returns (x, y) sorted ascending by x, where
        x = time after simulated impact (milliseconds)
        y = head acceleration (g)

    Source: Silverman (1985), via MASS::mcycle in R.
    """
    df = pd.read_csv(f"{_BASE}/MASS/mcycle.csv")
    x = df["times"].to_numpy(dtype=float)
    y = df["accel"].to_numpy(dtype=float)
    idx = np.argsort(x, kind="stable")
    return x[idx], y[idx]
