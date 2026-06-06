# Section B: Tabulate performance matrices
results = {"^NDX": ndx_result, "^GSPC": gspc_result}

print("MODEL PERFORMANCE MATRIX (IS: 1995-2021 | OOS: 2022-Jun 2026)")
print("="*72)
hdr = f"{'Metric':<42} {'^NDX':>12} {'GSPC':>12}"
print(hdr)
print("-"*72)

rows = [
    ("IS calibration window",        "1995-01-01",         "1995-01-01"),
    ("IS window end",                 "2021-12-31",         "2021-12-31"),
    ("OOS window start",              "2022-01-01",         "2022-01-01"),
    ("OOS observations",              str(ndx_result.oos_n_obs), str(gspc_result.oos_n_obs)),
    ("IS CF VaR 99% (1-day)",
        f"{ndx_result.is_var*100:.3f}%", f"{gspc_result.is_var*100:.3f}%"),
    ("OOS actual breaches",
        str(ndx_result.oos_n_breaches), str(gspc_result.oos_n_breaches)),
    ("OOS breach rate",
        f"{ndx_result.breach_rate*100:.2f}%", f"{gspc_result.breach_rate*100:.2f}%"),
    ("Theoretical breach rate",       "1.00%",              "1.00%"),
    ("Kupiec POF LR stat",
        f"{ndx_result.kupiec['lr_stat']:.3f}", f"{gspc_result.kupiec['lr_stat']:.3f}"),
    ("Kupiec POF p-value",
        f"{ndx_result.kupiec['pvalue']:.4f}", f"{gspc_result.kupiec['pvalue']:.4f}"),
    ("Kupiec POF reject H0 (5%)",
        "YES" if ndx_result.kupiec['reject_h0'] else "NO",
        "YES" if gspc_result.kupiec['reject_h0'] else "NO"),
    ("Christoffersen Ind LR",
        f"{ndx_result.christoffersen['lr_stat']:.3f}",
        f"{gspc_result.christoffersen['lr_stat']:.3f}"),
    ("Christoffersen p-value",
        f"{ndx_result.christoffersen['pvalue']:.4f}",
        f"{gspc_result.christoffersen['pvalue']:.4f}"),
    ("Christoffersen reject H0 (5%)",
        "YES" if ndx_result.christoffersen['reject_h0'] else "NO",
        "YES" if gspc_result.christoffersen['reject_h0'] else "NO"),
    ("Combined CC LR",
        f"{ndx_result.combined['lr_combined']:.3f}",
        f"{gspc_result.combined['lr_combined']:.3f}"),
    ("Combined CC p-value",
        f"{ndx_result.combined['pvalue_combined']:.4f}",
        f"{gspc_result.combined['pvalue_combined']:.4f}"),
    ("Mean IS-fold Kupiec p-value",
        f"{np.mean([k['pvalue'] for k in ndx_result.fold_kupiec]):.4f}",
        f"{np.mean([k['pvalue'] for k in gspc_result.fold_kupiec]):.4f}"),
]
for label, v1, v2 in rows:
    print(f"{label:<42} {v1:>12} {v2:>12}")
print("="*72)

# Drawdown velocity MAE
if ndx_result.fold_dd_vel_mae:
    print(f"\nDrawdown velocity MAE across IS folds:")
    print(f"  ^NDX  : {np.mean(ndx_result.fold_dd_vel_mae):.4f}%/day "
          f"(std={np.std(ndx_result.fold_dd_vel_mae):.4f})")
if gspc_result.fold_dd_vel_mae:
    print(f"  ^GSPC : {np.mean(gspc_result.fold_dd_vel_mae):.4f}%/day "
          f"(std={np.std(gspc_result.fold_dd_vel_mae):.4f})")
