# Section A: Load returns and run full IS/OOS backtest
ndx_raw  = pd.read_parquet(CACHE_DIR / "yfinance/idx_NDX_full_1995-01-01_2026-06-01.parquet")
gspc_raw = pd.read_parquet(CACHE_DIR / "yfinance/idx_GSPC_full_1995-01-01_2026-06-01.parquet")

ndx_returns  = np.log(ndx_raw["Close"]  / ndx_raw["Close"].shift(1)).dropna().rename("NDX")
gspc_returns = np.log(gspc_raw["Close"] / gspc_raw["Close"].shift(1)).dropna().rename("GSPC")

partitioner = TimeSeriesPartitioner()
backtester  = VaRBacktester(
    partitioner  = partitioner,
    var_fn       = calculate_cornish_fisher_var,
    confidence   = 0.99,
    n_cv_splits  = 5,
)

print("Running NDX backtest...")
ndx_result  = backtester.run(ndx_returns,  index_name="^NDX")
print("\nRunning GSPC backtest...")
gspc_result = backtester.run(gspc_returns, index_name="^GSPC")

print("\n" + "="*60)
print(ndx_result.summary())
print()
print(gspc_result.summary())
