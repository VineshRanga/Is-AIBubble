# Final validation assertions
print("Running Module 2 backtest assertions...")

# 1. IS/OOS never overlap
try:
    partitioner._check_no_overlap(
        ndx_returns.loc[IS_START:IS_END].index,
        ndx_returns.loc[OOS_START:OOS_END].index,
    )
    print("  PASS  IS/OOS windows have zero overlap (no look-ahead leakage)")
except ValueError as e:
    raise AssertionError(f"Look-ahead leakage detected: {e}")

# 2. OOS starts after IS ends
assert pd.Timestamp(OOS_START) > pd.Timestamp(IS_END), "OOS must start after IS ends"
print(f"  PASS  Chronological order enforced: IS ends {IS_END}  OOS starts {OOS_START}")

# 3. Reasonable number of OOS observations (at least 500 trading days)
for res in [ndx_result, gspc_result]:
    assert res.oos_n_obs >= 500, f"{res.index_name}: too few OOS observations ({res.oos_n_obs})"
print(f"  PASS  OOS observations sufficient: NDX={ndx_result.oos_n_obs}  GSPC={gspc_result.oos_n_obs}")

# 4. IS VaR is positive and plausible (between 0.5% and 10%)
for res in [ndx_result, gspc_result]:
    assert 0.005 <= res.is_var <= 0.10, (
        f"{res.index_name} IS VaR out of plausible range: {res.is_var*100:.3f}%"
    )
print(f"  PASS  IS CF VaR plausible: NDX={ndx_result.is_var*100:.3f}%  GSPC={gspc_result.is_var*100:.3f}%")

# 5. Breach count is non-negative and <= total OOS obs
for res in [ndx_result, gspc_result]:
    assert 0 <= res.oos_n_breaches <= res.oos_n_obs
print(f"  PASS  Breach counts valid: NDX={ndx_result.oos_n_breaches}  GSPC={gspc_result.oos_n_breaches}")

# 6. Statistical test stats are non-negative (chi-squared LR >= 0)
for res in [ndx_result, gspc_result]:
    assert res.kupiec["lr_stat"]       >= 0, "Kupiec LR must be >= 0"
    assert res.christoffersen["lr_stat"] >= 0, "Christoffersen LR must be >= 0"
    assert 0 <= res.kupiec["pvalue"]   <= 1, "p-value out of [0,1]"
print("  PASS  All test statistics are valid (LR >= 0, p in [0,1])")

# 7. Figures saved
for fname in ["fig9_var_backtest_oos.png", "fig10_drawdown_velocity_error.png"]:
    fpath = FIGURES_DIR / fname
    assert fpath.exists() and fpath.stat().st_size > 10_000, f"{fname} missing or too small"
    print(f"  PASS  {fname}  ({fpath.stat().st_size // 1024}KB)")

print("\nAll backtest assertions passed.")
