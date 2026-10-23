"""Calibration utilities for binary classifier probability estimates.

Ported from caloxtreme_clf/calibration/__init__.py.  Provides:

- ``expected_calibration_error`` — ECE metric
- ``TemperatureCalibrator`` — fit-once, apply-to-many temperature scaling
- ``calibrate_platt`` — Platt scaling (logistic regression on logits)
- ``compare_calibration_methods`` — driver that evaluates all methods

For MCMC we primarily use temperature scaling because it preserves the
rank ordering of density ratios and requires storing only a single float.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════
# 1.  Expected Calibration Error (ECE)
# ═══════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 100,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into *n_bins* equal-width bins across [0, 1],
    then computes the weighted average of |accuracy - confidence| per bin.

    Parameters
    ----------
    y_true : np.ndarray  shape (N,)
        Binary ground-truth labels.
    y_pred : np.ndarray  shape (N,)
        Predicted probabilities in [0, 1].
    n_bins : int
        Number of equal-width bins (default 100 for fine-grained ECE).

    Returns
    -------
    float
        ECE value in [0, 1].  Lower is better calibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        else:
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (n_bin / n_total) * abs(bin_acc - bin_conf)

    return float(ece)


# ═══════════════════════════════════════════════════════════════════════
# 0.  Abstract base calibrator
# ═══════════════════════════════════════════════════════════════════════

class BaseCalibrator(ABC):
    """Abstract base for classifier probability calibrators.

    Subclasses must implement ``transform_logits``, ``save``, and a
    ``load`` classmethod.  The unified contract is that
    ``transform_logits`` accepts raw classifier logits and returns
    calibrated logits such that ``sigmoid(calibrated_logits)`` gives
    well-calibrated probabilities.
    """

    @abstractmethod
    def transform_logits(self, logits):
        """Convert raw logits to calibrated logits.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor
            Raw logits from the classifier.

        Returns
        -------
        np.ndarray or torch.Tensor (same type as input)
            Calibrated logits.
        """
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the calibrator has been fitted."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save calibrator to file."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> "BaseCalibrator":
        """Factory: load a calibrator from file, auto-detecting method.

        Reads the file, inspects the ``"method"`` key (or infers
        ``"temperature"`` for legacy files), and returns the appropriate
        subclass instance.
        """
        path = Path(path)

        if path.suffix == ".npz":
            from scipy.interpolate import interp1d

            data = np.load(path)
            X_thresh = data["X_thresholds"]
            y_thresh = data["y_thresholds"]
            iso = IsotonicCalibrator()
            iso._iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso._iso_reg.X_thresholds_ = X_thresh
            iso._iso_reg.y_thresholds_ = y_thresh
            iso._iso_reg.X_min_ = float(X_thresh[0])
            iso._iso_reg.X_max_ = float(X_thresh[-1])
            iso._iso_reg.y_min = float(y_thresh.min())
            iso._iso_reg.y_max = float(y_thresh.max())
            iso._iso_reg.f_ = interp1d(
                X_thresh, y_thresh, kind="linear",
                bounds_error=False,
                fill_value=(y_thresh[0], y_thresh[-1]),
            )
            iso._X_thresholds = torch.from_numpy(X_thresh).float()
            iso._y_thresholds = torch.from_numpy(y_thresh).float()
            iso._use_torch = bool(data.get("use_torch", True))
            iso._fitted = True
            return iso

        with open(path) as f:
            data = json.load(f)

        method = data.get("method", "temperature")
        if method == "temperature":
            return TemperatureCalibrator(T=float(data["T"]))
        elif method == "platt":
            return PlattCalibrator(a=float(data["a"]), b=float(data["b"]))
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.is_fitted})"


# ═══════════════════════════════════════════════════════════════════════
# 2.  Temperature scaling calibrator
# ═══════════════════════════════════════════════════════════════════════

class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration (Guo et al. 2017).

    Learns a single temperature parameter *T* on validation data by
    minimising negative log-likelihood, then applies it to new scores:

        D_cal(x) = σ( logit(x) / T )

    *T* > 1 softens overconfident predictions; *T* < 1 sharpens
    underconfident ones.  *T* = 1 leaves the model unchanged.

    Parameters
    ----------
    T : float, optional
        Pre-fitted temperature.  If *None*, call ``fit()`` first.
    """

    def __init__(self, T: float | None = None):
        self.T: float | None = T

    @property
    def is_fitted(self) -> bool:
        return self.T is not None

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
        T_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> "TemperatureCalibrator":
        """Fit *T* on a held-out validation set.

        Parameters
        ----------
        yhat_val : np.ndarray  shape (N_val,)
            Raw predicted probabilities (before calibration) on validation set.
        y_val : np.ndarray  shape (N_val,)
            True binary labels for validation set.
        T_bounds : tuple[float, float]
            (lower, upper) bounds for the scalar optimisation.

        Returns
        -------
        self
        """
        logit_val = _prob_to_logit(yhat_val)
        eps = 1e-8

        def nll(T: float) -> float:
            t = max(T, 1e-6)
            calibrated = 1.0 / (1.0 + np.exp(-logit_val / t))
            calibrated = np.clip(calibrated, eps, 1.0 - eps)
            return -float(np.mean(
                y_val * np.log(calibrated)
                + (1.0 - y_val) * np.log(1.0 - calibrated)
            ))

        result = minimize_scalar(nll, bounds=T_bounds, method="bounded")
        self.T = float(result.x)
        return self

    # ------------------------------------------------------------------
    #  Apply
    # ------------------------------------------------------------------

    def transform(self, yhat: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw probabilities.

        Parameters
        ----------
        yhat : np.ndarray  shape (N,)
            Raw predicted probabilities.

        Returns
        -------
        np.ndarray  shape (N,)
            Calibrated probabilities.
        """
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        logits = _prob_to_logit(yhat)
        calibrated = 1.0 / (1.0 + np.exp(-logits / self.T))
        return np.clip(calibrated, 1e-8, 1.0 - 1e-8).astype(np.float64)

    def transform_logits(self, logits: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Apply temperature scaling directly to logits.

        This is the preferred method during MCMC since the classifier
        outputs raw logits and we want to avoid the double-sigmoid.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor  shape (N, 1) or (N,)
            Raw logits from the classifier.

        Returns
        -------
        Same type as input.
            Calibrated logits (divided by T).
        """
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return logits / self.T

    # ------------------------------------------------------------------
    #  Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save T to a JSON file."""
        if self.T is None:
            raise RuntimeError("Calibrator not fitted. Nothing to save.")
        with open(path, "w") as f:
            json.dump({"method": "temperature", "T": self.T}, f)

    def __repr__(self) -> str:
        t_str = f"{self.T:.4f}" if self.T is not None else "unfitted"
        return f"TemperatureCalibrator(T={t_str})"


# ═══════════════════════════════════════════════════════════════════════
# 2b.  Platt scaling calibrator
# ═══════════════════════════════════════════════════════════════════════

class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibration (Platt 1999).

    Fits a logistic regression on validation logits::

        D_cal(x) = σ(a · logit(x) + b)

    At inference, ``transform_logits`` returns ``a * logits + b``, which
    is directly usable as calibrated logits.

    Parameters
    ----------
    a : float, optional
        Logistic regression coefficient (slope).
    b : float, optional
        Logistic regression intercept.
    """

    def __init__(self, a: float | None = None, b: float | None = None):
        self._a = a
        self._b = b

    @property
    def is_fitted(self) -> bool:
        return self._a is not None and self._b is not None

    @property
    def a(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        return self._a

    @property
    def b(self) -> float:
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        return self._b

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "PlattCalibrator":
        """Fit Platt scaling on validation data.

        Parameters
        ----------
        yhat_val : np.ndarray  shape (N_val,)
            Raw predicted probabilities on validation set.
        y_val : np.ndarray  shape (N_val,)
            True binary labels.

        Returns
        -------
        self
        """
        logit_val = _prob_to_logit(yhat_val).reshape(-1, 1)

        lr = LogisticRegression(C=np.inf, solver="lbfgs")
        lr.fit(logit_val, y_val)

        self._a = float(lr.coef_[0, 0])
        self._b = float(lr.intercept_[0])
        return self

    def transform_logits(
        self, logits: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Apply Platt scaling to logits: ``a * logits + b``.

        Parameters
        ----------
        logits : np.ndarray or torch.Tensor
            Raw classifier logits.

        Returns
        -------
        Same type as input.
            Calibrated logits.
        """
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted. Call fit() first.")
        return self._a * logits + self._b

    def save(self, path: str | Path) -> None:
        """Save to JSON."""
        if not self.is_fitted:
            raise RuntimeError("PlattCalibrator not fitted.")
        with open(path, "w") as f:
            json.dump({"method": "platt", "a": self._a, "b": self._b}, f)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Isotonic regression calibrator
# ═══════════════════════════════════════════════════════════════════════

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration.

    Fits a non-parametric monotonic function on validation probabilities.
    Supports two backends:

    - **Torch-native** (default): pure-PyTorch piecewise linear
      interpolation using ``X_thresholds_`` / ``y_thresholds_``.  Stays
      on GPU, zero CPU transfers.
    - **sklearn fallback**: uses the fitted ``IsotonicRegression``
      object for correctness reference.

    At inference, ``transform_logits`` does::

        probs = σ(logits)
        cal_probs = isotonic_fn(probs)
        return logit(cal_probs)

    Parameters
    ----------
    use_torch : bool
        If True (default), use the torch-native interpolation path.
        If False, use sklearn ``.transform()``.
    """

    def __init__(self, use_torch: bool = True):
        self._iso_reg = None
        self._X_thresholds = None
        self._y_thresholds = None
        self._use_torch = use_torch
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        yhat_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression on validation probabilities."""
        self._iso_reg = IsotonicRegression(out_of_bounds="clip")
        self._iso_reg.fit(yhat_val.ravel(), y_val.ravel())

        # Extract thresholds for torch-native path
        self._X_thresholds = torch.from_numpy(
            self._iso_reg.X_thresholds_.copy()
        ).float()
        self._y_thresholds = torch.from_numpy(
            self._iso_reg.y_thresholds_.copy()
        ).float()
        self._fitted = True

        # Validate torch-native against sklearn on a small probe set
        if self._use_torch and len(yhat_val) > 0:
            probe = torch.linspace(0.0, 1.0, 101)
            torch_out = self._isotonic_transform_torch(probe).numpy()
            sklearn_out = self._iso_reg.transform(probe.numpy().reshape(-1, 1)).ravel()
            max_err = np.abs(torch_out - sklearn_out).max()
            if max_err > 1e-5:
                import warnings
                warnings.warn(
                    f"IsotonicCalibrator: torch vs sklearn max error = {max_err:.2e}. "
                    f"Consider use_torch=False for this calibrator."
                )
        return self

    def transform_logits(
        self, logits: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Apply isotonic calibration to logits.

        Returns calibrated logits: ``logit( f(σ(logits)) )``.
        """
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator not fitted. Call fit() first.")

        # Work in torch space for consistency; convert numpy inputs
        is_numpy = isinstance(logits, np.ndarray)
        if is_numpy:
            logits = torch.from_numpy(np.asarray(logits, dtype=np.float64))

        probs = torch.sigmoid(logits)

        if self._use_torch:
            cal_probs = self._isotonic_transform_torch(probs)
        else:
            cal_probs = self._isotonic_transform_sklearn(probs)

        # Convert calibrated probabilities back to logits
        eps = 1e-8
        cal_probs = torch.clamp(cal_probs, eps, 1.0 - eps)
        calibrated_logits = torch.log(cal_probs / (1.0 - cal_probs))

        if is_numpy:
            return calibrated_logits.numpy()
        return calibrated_logits

    def _isotonic_transform_torch(self, probs: torch.Tensor) -> torch.Tensor:
        """Torch-native piecewise linear interpolation.

        Mirrors sklearn's IsotonicRegression.transform() using
        searchsorted + linear interpolation within each bin.
        """
        X = self._X_thresholds.to(probs.device)
        Y = self._y_thresholds.to(probs.device)
        n_thresh = len(X)

        if n_thresh == 0:
            return probs

        # Find the right bin for each probability
        idx = torch.searchsorted(X, probs)
        idx = idx.clamp(1, n_thresh - 1)

        x_lo = X[idx - 1]
        x_hi = X[idx]
        y_lo = Y[idx - 1]
        y_hi = Y[idx]

        # Linear interpolation: y = y_lo + (y_hi - y_lo) * (x - x_lo) / (x_hi - x_lo)
        t = (probs - x_lo) / (x_hi - x_lo + 1e-10)
        t = t.clamp(0.0, 1.0)
        return y_lo + t * (y_hi - y_lo)

    def _isotonic_transform_sklearn(self, probs: torch.Tensor) -> torch.Tensor:
        """sklearn fallback: CPU roundtrip through IsotonicRegression.transform."""
        probs_np = probs.cpu().numpy().reshape(-1, 1)
        cal_np = self._iso_reg.transform(probs_np).ravel()
        return torch.from_numpy(cal_np).to(probs.device)

    def save(self, path: str | Path) -> None:
        """Save thresholds to .npz."""
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator not fitted.")
        np.savez(
            path,
            X_thresholds=self._iso_reg.X_thresholds_,
            y_thresholds=self._iso_reg.y_thresholds_,
            use_torch=self._use_torch,
        )

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        """Load thresholds from .npz."""
        return BaseCalibrator.load(path)


# ═══════════════════════════════════════════════════════════════════════
# 4.  Platt scaling (alternative calibration — legacy)
# ═══════════════════════════════════════════════════════════════════════

def calibrate_platt(
    yhat_val: np.ndarray,
    y_val: np.ndarray,
    yhat_test: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Platt scaling calibration (Platt 1999).

    Fits a logistic regression model on the validation logits,
    then applies it to test scores.

    Parameters
    ----------
    yhat_val : np.ndarray  shape (N_val,)
        Raw predicted probabilities on validation set.
    y_val : np.ndarray  shape (N_val,)
        True labels on validation set.
    yhat_test : np.ndarray  shape (N_test,)
        Raw predicted probabilities to calibrate.

    Returns
    -------
    calibrated : np.ndarray  shape (N_test,)
    metadata : dict  with keys "method", "coef", "intercept"
    """
    logit_val = _prob_to_logit(yhat_val).reshape(-1, 1)
    logit_test = _prob_to_logit(yhat_test).reshape(-1, 1)

    lr = LogisticRegression(C=np.inf, solver="lbfgs")
    lr.fit(logit_val, y_val)

    calibrated = lr.predict_proba(logit_test)[:, 1]
    metadata = {
        "method": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }
    return calibrated, metadata


# ═══════════════════════════════════════════════════════════════════════
# 5.  Method comparison driver
# ═══════════════════════════════════════════════════════════════════════

def compare_calibration_methods(
    yhat_val: np.ndarray,
    y_val: np.ndarray,
    yhat_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Compare raw, isotonic, temperature, and Platt scaling on test set.

    Parameters
    ----------
    yhat_val, yhat_test : np.ndarray
        Raw predictions on validation / test sets.
    y_val, y_test : np.ndarray
        True labels.
    n_bins : int
        Number of bins for ECE.

    Returns
    -------
    dict
        Method name → {"ece": float, "brier": float, "scores": np.ndarray, ...}
    """
    from sklearn.metrics import brier_score_loss

    results: dict = {}

    # Raw
    results["raw"] = {
        "ece": expected_calibration_error(y_test, yhat_test, n_bins),
        "brier": float(brier_score_loss(y_test, yhat_test)),
        "scores": yhat_test,
    }

    # Isotonic
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(yhat_val, y_val)
    cal_iso = ir.transform(yhat_test)
    results["isotonic"] = {
        "ece": expected_calibration_error(y_test, cal_iso, n_bins),
        "brier": float(brier_score_loss(y_test, cal_iso)),
        "scores": cal_iso,
    }

    # Temperature
    calib = TemperatureCalibrator()
    calib.fit(yhat_val, y_val)
    cal_temp = calib.transform(yhat_test)
    results["temperature"] = {
        "ece": expected_calibration_error(y_test, cal_temp, n_bins),
        "brier": float(brier_score_loss(y_test, cal_temp)),
        "scores": cal_temp,
        "T": calib.T,
    }

    # Platt
    cal_platt, platt_meta = calibrate_platt(yhat_val, y_val, yhat_test)
    results["platt"] = {
        "ece": expected_calibration_error(y_test, cal_platt, n_bins),
        "brier": float(brier_score_loss(y_test, cal_platt)),
        "scores": cal_platt,
        "coef": platt_meta["coef"],
        "intercept": platt_meta["intercept"],
    }

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6.  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _prob_to_logit(prob: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert probability scores to logits (inverse sigmoid)."""
    prob = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(prob / (1.0 - prob))


# Import needed for transform_logits type hint
import torch  # noqa: E402
