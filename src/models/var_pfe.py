"""
src/models/var_pfe.py
---------------------
Tail-risk engines for Module 2: Value at Risk (VaR), Potential Future Exposure
(PFE), and scenario stress testing under the 2026 IPO liquidity vacuum shock.

References
----------
- Cornish-Fisher VaR expansion: Favre & Galeano (2002), J. Alternative Investments.
- PFE estimation: Basel III Annex IV, counterparty credit risk framework.
- Historical simulation VaR: Hull (2018), Risk Management and Financial Institutions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Historical Simulation VaR ─────────────────────────────────────────────────

def calculate_historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    window: int = 252,
) -> float:
    """
    Standard historical simulation Value at Risk.

    Uses a rolling tail of the most recent `window` observations to estimate
    the loss threshold exceeded with probability (1 - confidence).

    Parameters
    ----------
    returns    : pd.Series of daily log-returns (negatives = losses).
    confidence : VaR confidence level (e.g. 0.99 = 99%).
    window     : Number of most-recent trading days to include.

    Returns
    -------
    float
        VaR as a positive number representing the maximum expected 1-day
        loss at the given confidence level (e.g. 0.023 = 2.3%).
    """
    if len(returns) == 0:
        raise ValueError("returns series is empty")
    tail = returns.iloc[-window:] if len(returns) >= window else returns
    var = float(-np.percentile(tail.dropna(), 100 * (1 - confidence)))
    logger.debug("Historical VaR(%.0f%%, n=%d) = %.4f", confidence * 100, len(tail), var)
    return var


# ── Cornish-Fisher VaR ────────────────────────────────────────────────────────

def calculate_cornish_fisher_var(
    returns: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Parametric VaR adjusted for skewness and excess kurtosis (Cornish-Fisher).

    The standard Gaussian quantile z is adjusted by the modified z_CF:

        z_CF = z + (z²-1)*S/6 + (z³-3z)*K/24 - (2z³-5z)*S²/36

    where S = skewness and K = excess kurtosis of the return distribution.
    This is essential for fat-tailed tech index drawdowns where the Gaussian
    model systematically underestimates tail risk.

    Parameters
    ----------
    returns    : pd.Series of daily log-returns.
    confidence : VaR confidence level.

    Returns
    -------
    float
        Cornish-Fisher VaR as a positive loss magnitude.
    """
    clean = returns.dropna()
    if len(clean) < 20:
        raise ValueError(f"Insufficient observations for CF-VaR: {len(clean)} < 20")

    mu    = float(clean.mean())
    sigma = float(clean.std(ddof=1))
    S     = float(stats.skew(clean))           # skewness
    K     = float(stats.kurtosis(clean))       # excess kurtosis (Fisher)

    z     = stats.norm.ppf(1 - confidence)     # Gaussian quantile (negative)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * S / 6
        + (z**3 - 3 * z) * K / 24
        - (2 * z**3 - 5 * z) * S**2 / 36
    )

    var = -(mu + sigma * z_cf)
    logger.debug(
        "CF-VaR(%.0f%%): mu=%.4f sigma=%.4f S=%.3f K=%.3f z=%.3f z_cf=%.3f VaR=%.4f",
        confidence * 100, mu, sigma, S, K, z, z_cf, var,
    )
    return float(var)


# ── Potential Future Exposure ─────────────────────────────────────────────────

def calculate_pfe(
    simulated_paths: np.ndarray,
    confidence: float = 0.99,
    base_price: Optional[float] = None,
) -> np.ndarray:
    """
    Potential Future Exposure (PFE) at each time step of a Monte Carlo simulation.

    For each column (time step t), PFE[t] is the (1-confidence) percentile of
    the simulated loss distribution relative to the initial price level. This
    gives the worst-case expected loss under the stress scenario at each horizon.

    Parameters
    ----------
    simulated_paths : np.ndarray of shape (n_paths, n_timesteps).
                      Each row is one price path; each column one time step.
    confidence      : PFE confidence level (e.g. 0.99 = 99th percentile).
    base_price      : Reference price for computing % loss. If None, uses
                      the first time step's median across paths.

    Returns
    -------
    np.ndarray of shape (n_timesteps,)
        PFE expressed as a positive fraction of the base price (e.g. 0.35 = 35% loss).
    """
    if simulated_paths.ndim == 1:
        simulated_paths = simulated_paths.reshape(1, -1)

    n_paths, n_steps = simulated_paths.shape
    if base_price is None:
        base_price = float(np.median(simulated_paths[:, 0]))

    pfe = np.empty(n_steps)
    for t in range(n_steps):
        losses_t = (base_price - simulated_paths[:, t]) / base_price
        pfe[t]   = float(np.percentile(losses_t, 100 * confidence))

    logger.debug("PFE computed over %d paths x %d steps at %.0f%% confidence",
                 n_paths, n_steps, confidence * 100)
    return pfe


# ── Scenario Stress Tester ────────────────────────────────────────────────────

@dataclass
class ScenarioStressTester:
    """
    Applies a macro stress scenario to a portfolio/index and computes stressed
    VaR and PFE using the 2022 correlation matrix as the calibrated regime.

    Parameters
    ----------
    index_returns : pd.DataFrame
        Daily log-returns indexed by date. Each column is one asset.
        For single-index analysis pass a DataFrame with one column.
    weights : dict[str, float] | None
        Portfolio weights. If None, assumes equal weighting across columns.
    calibration_start : str
        Start of the stress calibration window (default: 2022-01-01).
    calibration_end : str
        End of the stress calibration window (default: 2022-12-31).

    Examples
    --------
    >>> tester = ScenarioStressTester(index_returns=ndx_returns_df)
    >>> result = tester.apply_shock({"NDX": -0.30}, ipo_supply_b=210.0)
    >>> print(result.stressed_cf_var_99)
    """

    index_returns:     pd.DataFrame
    weights:           Optional[dict[str, float]] = None
    calibration_start: str = "2022-01-01"
    calibration_end:   str = "2022-12-31"

    # Populated after fit()
    _corr_matrix:  pd.DataFrame = field(default=None, repr=False, init=False)
    _vol_vector:   pd.Series    = field(default=None, repr=False, init=False)
    _fitted:       bool         = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.weights is None:
            n = len(self.index_returns.columns)
            self.weights = {c: 1.0 / n for c in self.index_returns.columns}
        self._validate()

    def _validate(self) -> None:
        missing = [k for k in self.weights if k not in self.index_returns.columns]
        if missing:
            raise ValueError(f"Weight keys not in returns columns: {missing}")

    def fit(self) -> "ScenarioStressTester":
        """
        Fit correlation matrix and volatilities from the 2022 calibration window.

        Returns self for method chaining.
        """
        cal = self.index_returns.loc[
            self.calibration_start : self.calibration_end
        ].dropna()
        if len(cal) < 20:
            logger.warning(
                "Calibration window has only %d observations; results may be unreliable.", len(cal)
            )
        self._corr_matrix = cal.corr()
        self._vol_vector  = cal.std(ddof=1) * np.sqrt(252)   # annualised
        self._fitted      = True
        logger.info(
            "StressTester fitted on %d observations (%s to %s)",
            len(cal), self.calibration_start, self.calibration_end,
        )
        return self

    @property
    def corr_matrix(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing calibration outputs.")
        return self._corr_matrix

    @property
    def annualised_vols(self) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Call .fit() before accessing calibration outputs.")
        return self._vol_vector

    def apply_shock(
        self,
        shock_multipliers: dict[str, float],
        ipo_supply_b: float = 0.0,
        mlb_b: float = 19_000.0,
        n_paths: int = 2_000,
        horizon_days: int = 252,
        seed: int = 42,
    ) -> "StressResult":
        """
        Apply a scenario shock to the portfolio and simulate stressed paths.

        The total shock combines two components:
        1. **Structural shock** from `shock_multipliers` — a deterministic total
           log-return applied linearly over `horizon_days`.
        2. **Liquidity shock** — an additional impulse derived from the IPO
           supply pressure ratio (ipo_supply_b / mlb_b), scaled by the
           Cornish-Fisher VaR of the calibration window.

        Parameters
        ----------
        shock_multipliers : dict mapping column name to total log-return shock
                            (e.g. {"NDX": -0.30} means -30% deterministic drift).
        ipo_supply_b      : IPO float supply in $B (applied as liquidity impulse).
        mlb_b             : Market Liquidity Buffer estimate in $B.
        n_paths           : Number of Monte Carlo simulation paths.
        horizon_days      : Forward simulation horizon in trading days.
        seed              : RNG seed for reproducibility.

        Returns
        -------
        StressResult
            Dataclass containing stressed VaR, PFE array, and simulated paths.
        """
        if not self._fitted:
            self.fit()

        rng      = np.random.default_rng(seed)
        col      = list(self.index_returns.columns)[0]
        daily_r  = self.index_returns[col].dropna()
        sigma_d  = float(daily_r.std(ddof=1))

        struct_lr  = shock_multipliers.get(col, 0.0)
        struct_d   = struct_lr / horizon_days

        # Liquidity impulse: front-loaded over first 126 days
        supply_pressure = ipo_supply_b / max(mlb_b, 1.0)
        liq_impulse_d   = -(supply_pressure * abs(struct_lr) * 0.5) / min(126, horizon_days)
        liq_arr         = np.zeros(horizon_days)
        liq_arr[:min(126, horizon_days)] = liq_impulse_d

        paths = np.empty((n_paths, horizon_days))
        for p in range(n_paths):
            noise   = rng.normal(0, sigma_d, horizon_days)
            daily   = struct_d + noise + liq_arr
            paths[p] = np.exp(np.cumsum(daily))   # normalised to S0=1

        cf_var  = calculate_cornish_fisher_var(daily_r, confidence=0.99)
        hist_var = calculate_historical_var(daily_r, confidence=0.99)
        pfe     = calculate_pfe(paths, confidence=0.99, base_price=1.0)

        return StressResult(
            scenario_name          = str(shock_multipliers),
            structural_shock       = struct_lr,
            supply_pressure        = supply_pressure,
            hist_var_99            = hist_var,
            cf_var_99              = cf_var,
            pfe_curve              = pfe,
            simulated_paths        = paths,
            horizon_days           = horizon_days,
        )

    def compute_stressed_var(
        self,
        shock: float,
        confidence: float = 0.99,
    ) -> float:
        """
        Compute Cornish-Fisher VaR on returns scaled by the structural shock.

        Parameters
        ----------
        shock      : Total log-return shock to apply (e.g. -0.30).
        confidence : VaR confidence level.

        Returns
        -------
        float
            CF-VaR expressed as a positive loss fraction.
        """
        col     = list(self.index_returns.columns)[0]
        daily_r = self.index_returns[col].dropna()
        # Shift the distribution by the daily shock equivalent
        n_days  = 252
        shifted = daily_r + shock / n_days
        return calculate_cornish_fisher_var(shifted, confidence=confidence)


@dataclass
class StressResult:
    """Container for ScenarioStressTester output."""

    scenario_name:    str
    structural_shock: float
    supply_pressure:  float
    hist_var_99:      float
    cf_var_99:        float
    pfe_curve:        np.ndarray
    simulated_paths:  np.ndarray
    horizon_days:     int

    def peak_pfe(self) -> float:
        """Maximum PFE across the simulation horizon."""
        return float(self.pfe_curve.max())

    def pfe_at_day(self, day: int) -> float:
        """PFE at a specific horizon day."""
        idx = min(day, self.horizon_days - 1)
        return float(self.pfe_curve[idx])

    def summary(self) -> str:
        return (
            f"Scenario : {self.scenario_name}\n"
            f"Shock    : {self.structural_shock * 100:+.1f}%\n"
            f"SP ratio : {self.supply_pressure:.5f}\n"
            f"Hist VaR (99%): {self.hist_var_99 * 100:.2f}%\n"
            f"CF   VaR (99%): {self.cf_var_99 * 100:.2f}%\n"
            f"Peak PFE (99%): {self.peak_pfe() * 100:.2f}%  "
            f"(day {int(self.pfe_curve.argmax())})\n"
        )
