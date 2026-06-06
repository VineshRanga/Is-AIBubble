"""
src/models/validation.py
------------------------
Look-ahead-free time-series validation, VaR backtesting, and statistical tests
for the AI Capex macro risk research project.

Chronological split boundary (hard-coded, project-wide):
  In-Sample  (IS) : 1995-01-01 through 2021-12-31
  Out-of-Sample   : 2022-01-01 through 2026-06-01

Statistical tests implemented:
  - Kupiec (1995) Proportion of Failures (POF) test
  - Christoffersen (1998) Interval Independence test
  - Combined Christoffersen conditional-coverage test

References
----------
- Kupiec, P.H. (1995). Techniques for Verifying the Accuracy of Risk Measurement
  Models. Journal of Derivatives, 3(2), 73–84.
- Christoffersen, P.F. (1998). Evaluating Interval Forecasts. International
  Economic Review, 39(4), 841–862.
- Lopez, J.A. (1998). Methods for Evaluating Value-at-Risk Estimates.
  Federal Reserve Bank of New York Economic Policy Review, 4(3), 119–124.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# ── Project-wide chronological boundaries ────────────────────────────────────
IS_START:   str = "1995-01-01"
IS_END:     str = "2021-12-31"
OOS_START:  str = "2022-01-01"
OOS_END:    str = "2026-06-01"


# ── TimeSeriesPartitioner ─────────────────────────────────────────────────────

class TimeSeriesPartitioner:
    """
    Enforces strict chronological IS/OOS partitioning with no look-ahead leakage.

    All features derived from rolling or expanding windows are computed using
    only historical data up to and including time t — no access to t+1.

    Parameters
    ----------
    is_start  : IS calibration start date (inclusive).
    is_end    : IS calibration end date (inclusive). All training data must end here.
    oos_start : OOS evaluation start date (inclusive).
    oos_end   : OOS evaluation end date (inclusive).
    """

    def __init__(
        self,
        is_start:  str = IS_START,
        is_end:    str = IS_END,
        oos_start: str = OOS_START,
        oos_end:   str = OOS_END,
    ) -> None:
        self.is_start  = pd.Timestamp(is_start)
        self.is_end    = pd.Timestamp(is_end)
        self.oos_start = pd.Timestamp(oos_start)
        self.oos_end   = pd.Timestamp(oos_end)

    def split_series(
        self, series: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Partition a time series into IS and OOS windows.

        Returns
        -------
        (is_data, oos_data) — both as pd.Series slices.
        """
        is_data  = series.loc[self.is_start  : self.is_end]
        oos_data = series.loc[self.oos_start : self.oos_end]
        self._check_no_overlap(is_data.index, oos_data.index)
        logger.debug(
            "IS: %d obs (%s→%s)  OOS: %d obs (%s→%s)",
            len(is_data),  str(is_data.index.min().date()), str(is_data.index.max().date()),
            len(oos_data), str(oos_data.index.min().date()), str(oos_data.index.max().date()),
        )
        return is_data, oos_data

    @staticmethod
    def _check_no_overlap(idx_is: pd.DatetimeIndex, idx_oos: pd.DatetimeIndex) -> None:
        overlap = idx_is.intersection(idx_oos)
        if len(overlap):
            raise ValueError(
                f"IS/OOS overlap detected on {len(overlap)} dates — "
                f"look-ahead leakage possible. First overlap: {overlap[0]}"
            )

    def tscv_folds(
        self,
        series: pd.Series,
        n_splits: int = 5,
    ) -> list[tuple[pd.Series, pd.Series]]:
        """
        Generate TimeSeriesSplit folds **entirely within the IS window**.

        Validation folds are guaranteed to end on or before `is_end`, so no
        fold can accidentally include post-2021 data.

        Returns
        -------
        list of (train_fold, val_fold) pairs.
        """
        is_data = series.loc[self.is_start : self.is_end]
        tscv    = TimeSeriesSplit(n_splits=n_splits)
        folds   = []
        for tr_idx, val_idx in tscv.split(is_data):
            train_fold = is_data.iloc[tr_idx]
            val_fold   = is_data.iloc[val_idx]
            folds.append((train_fold, val_fold))
        logger.debug("Generated %d IS-only TimeSeriesSplit folds", len(folds))
        return folds

    def expanding_feature(
        self,
        series: pd.Series,
        fn: Callable[[pd.Series], float],
        min_periods: int = 252,
    ) -> pd.Series:
        """
        Compute a scalar feature on a strictly expanding window ending at each t.

        Guarantees zero look-ahead: the feature at time t is computed using only
        observations up to and including t.

        Parameters
        ----------
        series      : Input time series.
        fn          : Function mapping a Series slice to a scalar.
        min_periods : Minimum observations before emitting a non-NaN value.

        Returns
        -------
        pd.Series aligned with the input index.
        """
        out = pd.Series(np.nan, index=series.index)
        for t in range(min_periods, len(series) + 1):
            out.iloc[t - 1] = fn(series.iloc[:t])
        return out


# ── Kupiec Proportion-of-Failures (POF) Test ─────────────────────────────────

def kupiec_pof_test(
    n_breaches: int,
    n_obs:      int,
    confidence: float = 0.99,
) -> dict[str, float]:
    """
    Kupiec (1995) Proportion-of-Failures test for VaR accuracy.

    Tests H0: observed breach frequency equals the theoretical VaR tail probability.
    Null hypothesis of correct VaR is rejected when the LR statistic exceeds the
    chi-squared critical value.

    Log-likelihood ratio:
        LR_POF = -2 * ln[(p^x * (1-p)^(T-x)) / (p_hat^x * (1-p_hat)^(T-x))]

    where T = number of observations, x = number of breaches, p = theoretical
    breach probability (1 - confidence), p_hat = x / T.

    Under H0: LR_POF ~ chi-squared(1).

    Parameters
    ----------
    n_breaches : Observed number of VaR breaches.
    n_obs      : Total number of observations.
    confidence : VaR confidence level (e.g. 0.99 for 99% VaR).

    Returns
    -------
    dict with keys: lr_stat, pvalue, n_breaches, n_obs, breach_rate,
                    theoretical_rate, reject_h0 (at 5% level).
    """
    p     = 1.0 - confidence    # theoretical breach probability
    x     = n_breaches
    T     = n_obs
    p_hat = x / T if T > 0 else 0.0

    if x == 0:
        # log(0) limit: LR -> -2 * T * log(1-p)
        lr = -2.0 * T * np.log(1.0 - p)
    elif x == T:
        lr = -2.0 * T * np.log(p)
    else:
        lr = -2.0 * (
            x * np.log(p / p_hat)
            + (T - x) * np.log((1.0 - p) / (1.0 - p_hat))
        )

    pvalue    = float(1.0 - stats.chi2.cdf(lr, df=1))
    reject_h0 = pvalue < 0.05

    return {
        "lr_stat":         float(lr),
        "pvalue":          pvalue,
        "n_breaches":      x,
        "n_obs":           T,
        "breach_rate":     p_hat,
        "theoretical_rate": p,
        "reject_h0":       reject_h0,
    }


# ── Christoffersen Interval Independence Test ──────────────────────────────────

def christoffersen_independence_test(
    breach_indicators: np.ndarray,
) -> dict[str, float]:
    """
    Christoffersen (1998) test for serial independence of VaR breaches.

    Tests H0: breach indicators form an i.i.d. Bernoulli sequence (no clustering).
    Clustering of breaches (common during stress regimes) indicates that the VaR
    model fails to capture autocorrelation in the tail loss distribution.

    Transition counts:
        n_ij = number of transitions from state i to state j
        (0 = no breach, 1 = breach)

    Log-likelihood ratio:
        LR_ind = 2 * ln[L(pi_01, pi_11) / L(pi_2)]

    where pi_01, pi_11 are estimated transition probabilities and pi_2 is the
    unconditional breach probability.

    Under H0: LR_ind ~ chi-squared(1).

    Parameters
    ----------
    breach_indicators : Binary array (1 = breach, 0 = no breach).

    Returns
    -------
    dict with keys: lr_stat, pvalue, pi_01, pi_11, pi_2, n_00, n_01, n_10, n_11,
                    reject_h0 (at 5% level).
    """
    I = np.asarray(breach_indicators, dtype=int)
    if len(I) < 2:
        raise ValueError("Need at least 2 observations for independence test.")

    n_00 = int(np.sum((I[:-1] == 0) & (I[1:] == 0)))
    n_01 = int(np.sum((I[:-1] == 0) & (I[1:] == 1)))
    n_10 = int(np.sum((I[:-1] == 1) & (I[1:] == 0)))
    n_11 = int(np.sum((I[:-1] == 1) & (I[1:] == 1)))

    pi_01 = n_01 / max(n_00 + n_01, 1)
    pi_11 = n_11 / max(n_10 + n_11, 1)
    pi_2  = (n_01 + n_11) / max(n_00 + n_01 + n_10 + n_11, 1)

    # Alternative (Markov) log-likelihood
    def safe_log(x: float) -> float:
        return np.log(max(x, 1e-300))

    l_alt = (
        safe_log(1 - pi_01) * n_00
        + safe_log(pi_01) * n_01
        + safe_log(1 - pi_11) * n_10
        + safe_log(pi_11) * n_11
    )
    l_null = (
        safe_log(1 - pi_2) * (n_00 + n_10)
        + safe_log(pi_2) * (n_01 + n_11)
    )

    lr        = float(2.0 * (l_alt - l_null))
    lr        = max(lr, 0.0)   # numerical guard
    pvalue    = float(1.0 - stats.chi2.cdf(lr, df=1))
    reject_h0 = pvalue < 0.05

    return {
        "lr_stat":   lr,
        "pvalue":    pvalue,
        "pi_01":     pi_01,
        "pi_11":     pi_11,
        "pi_2":      pi_2,
        "n_00":      n_00,
        "n_01":      n_01,
        "n_10":      n_10,
        "n_11":      n_11,
        "reject_h0": reject_h0,
    }


def christoffersen_combined_test(
    n_breaches: int,
    n_obs:      int,
    breach_indicators: np.ndarray,
    confidence: float = 0.99,
) -> dict[str, float]:
    """
    Combined Christoffersen conditional-coverage test.

    Combines the POF test and the independence test into a single joint test:
        LR_cc = LR_POF + LR_ind ~ chi-squared(2) under H0.

    H0: breach frequency is correct AND breaches are serially independent.

    Returns
    -------
    dict with pof, independence, and combined test results.
    """
    pof  = kupiec_pof_test(n_breaches, n_obs, confidence)
    indp = christoffersen_independence_test(breach_indicators)
    lr_cc     = pof["lr_stat"] + indp["lr_stat"]
    pvalue_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    return {
        "lr_pof":        pof["lr_stat"],
        "lr_ind":        indp["lr_stat"],
        "lr_combined":   lr_cc,
        "pvalue_pof":    pof["pvalue"],
        "pvalue_ind":    indp["pvalue"],
        "pvalue_combined": pvalue_cc,
        "reject_h0_pof":    pof["reject_h0"],
        "reject_h0_ind":    indp["reject_h0"],
        "reject_h0_combined": pvalue_cc < 0.05,
        "breach_rate":   pof["breach_rate"],
        "theoretical_rate": pof["theoretical_rate"],
    }


# ── Drawdown Velocity ─────────────────────────────────────────────────────────

def drawdown_velocity(
    returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """
    Compute rolling maximum drawdown velocity: maximum drawdown magnitude
    divided by the number of trading days from peak to trough, over a trailing
    window.

    Velocity is expressed as % per trading day (positive = rate of loss).

    Parameters
    ----------
    returns : pd.Series of daily log-returns.
    window  : Rolling window length in trading days.

    Returns
    -------
    pd.Series of annualised drawdown velocities aligned with the input index.
    """
    prices     = np.exp(returns.cumsum()).to_numpy()
    velocities = pd.Series(np.nan, index=returns.index)
    for t in range(window, len(prices)):
        sub      = prices[t - window : t]
        peak_pos = int(np.argmax(sub))
        rest     = sub[peak_pos:]
        trough_pos = peak_pos + int(np.argmin(rest))
        peak_v   = sub[peak_pos]
        trough_v = sub[trough_pos]
        max_dd   = (peak_v - trough_v) / max(peak_v, 1e-12)
        days     = max(trough_pos - peak_pos, 1)
        velocities.iloc[t - 1] = max_dd / days * 100.0   # % per day
    return velocities


# ── VaR Backtest Engine ───────────────────────────────────────────────────────

@dataclass
class VaRBacktestResult:
    """Output container for a full IS/OOS VaR backtest."""

    index_name:        str
    confidence:        float
    is_var:            float
    oos_n_breaches:    int
    oos_n_obs:         int
    breach_rate:       float
    kupiec:            dict
    christoffersen:    dict
    combined:          dict
    oos_returns:       pd.Series
    breach_mask:       pd.Series
    var_series:        pd.Series
    fold_kupiec:       list[dict] = field(default_factory=list)
    fold_dd_vel_mae:   list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== VaR Backtest: {self.index_name} ({self.confidence*100:.0f}%) ===",
            f"IS VaR (99%, CF)    : {self.is_var*100:.3f}%",
            f"OOS observations    : {self.oos_n_obs}",
            f"OOS breaches        : {self.oos_n_breaches}  "
              f"({self.breach_rate*100:.2f}% vs 1.00% theoretical)",
            f"Kupiec POF  LR={self.kupiec['lr_stat']:.3f}  "
              f"p={self.kupiec['pvalue']:.4f}  "
              f"reject={'YES' if self.kupiec['reject_h0'] else 'NO'}",
            f"Christoffersen Ind  LR={self.christoffersen['lr_stat']:.3f}  "
              f"p={self.christoffersen['pvalue']:.4f}  "
              f"reject={'YES' if self.christoffersen['reject_h0'] else 'NO'}",
            f"Combined CC  p={self.combined['pvalue_combined']:.4f}  "
              f"reject={'YES' if self.combined['reject_h0_combined'] else 'NO'}",
        ]
        return "\n".join(lines)


class VaRBacktester:
    """
    Run a full IS calibration → OOS evaluation backtest for a VaR model.

    Enforces strict look-ahead separation via TimeSeriesPartitioner.

    Parameters
    ----------
    partitioner  : Configured TimeSeriesPartitioner instance.
    var_fn       : Callable(pd.Series, confidence) -> float
                   Function that maps a return series to a VaR estimate.
    confidence   : VaR confidence level.
    n_cv_splits  : Number of TimeSeriesSplit folds for IS cross-validation.
    """

    def __init__(
        self,
        partitioner: TimeSeriesPartitioner,
        var_fn: Callable[[pd.Series, float], float],
        confidence: float = 0.99,
        n_cv_splits: int = 5,
    ) -> None:
        self.partitioner  = partitioner
        self.var_fn       = var_fn
        self.confidence   = confidence
        self.n_cv_splits  = n_cv_splits

    def run(
        self,
        returns: pd.Series,
        index_name: str = "Index",
    ) -> VaRBacktestResult:
        """
        Execute the full backtest pipeline.

        Steps:
        1. Partition into IS and OOS windows.
        2. Calibrate VaR on the full IS window.
        3. Apply the VaR to the OOS window; identify breaches.
        4. Run Kupiec POF, Christoffersen independence, and combined tests.
        5. Cross-validate within the IS window using TimeSeriesSplit folds.
        6. Compute drawdown velocity MAE per fold.

        Parameters
        ----------
        returns    : Full daily log-return series (IS + OOS combined).
        index_name : Label for reporting.

        Returns
        -------
        VaRBacktestResult
        """
        is_ret, oos_ret = self.partitioner.split_series(returns)

        # IS calibration
        is_var = self.var_fn(is_ret, self.confidence)
        logger.info("%s: IS VaR (%.0f%%) = %.4f", index_name, self.confidence * 100, is_var)

        # OOS breach detection: breach when daily loss exceeds VaR
        oos_loss    = -oos_ret                           # positive = loss
        breach_mask = oos_loss > is_var
        n_breaches  = int(breach_mask.sum())
        n_obs       = len(oos_ret)

        # VaR constant band for OOS (IS-calibrated, no updating)
        var_series = pd.Series(-is_var, index=oos_ret.index)

        kupiec_res  = kupiec_pof_test(n_breaches, n_obs, self.confidence)
        chrs_res    = christoffersen_independence_test(breach_mask.values)
        combined    = christoffersen_combined_test(n_breaches, n_obs, breach_mask.values, self.confidence)

        # IS cross-validation folds
        folds       = self.partitioner.tscv_folds(returns, n_splits=self.n_cv_splits)
        fold_kupiec = []
        fold_dd_vel_mae = []

        for train_fold, val_fold in folds:
            fold_var     = self.var_fn(train_fold, self.confidence)
            fold_loss    = -val_fold
            fold_breach  = int((fold_loss > fold_var).sum())
            fold_kp      = kupiec_pof_test(fold_breach, len(val_fold), self.confidence)
            fold_kupiec.append(fold_kp)

            # Drawdown velocity MAE
            if len(val_fold) >= 63:
                pred_vel  = fold_var * np.sqrt(63) * 100 / 63   # annualized daily vel proxy
                vel_series = drawdown_velocity(val_fold, window=63).dropna()
                if len(vel_series):
                    act_vel = float(vel_series.mean())
                    fold_dd_vel_mae.append(abs(pred_vel - act_vel))

        logger.info(
            "%s OOS: %d breaches / %d obs  (%.2f%%)  Kupiec p=%.4f",
            index_name, n_breaches, n_obs, n_breaches / n_obs * 100, kupiec_res["pvalue"],
        )

        return VaRBacktestResult(
            index_name      = index_name,
            confidence      = self.confidence,
            is_var          = is_var,
            oos_n_breaches  = n_breaches,
            oos_n_obs       = n_obs,
            breach_rate     = n_breaches / n_obs,
            kupiec          = kupiec_res,
            christoffersen  = chrs_res,
            combined        = combined,
            oos_returns     = oos_ret,
            breach_mask     = breach_mask,
            var_series      = var_series,
            fold_kupiec     = fold_kupiec,
            fold_dd_vel_mae = fold_dd_vel_mae,
        )
