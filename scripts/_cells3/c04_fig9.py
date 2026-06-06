# Figure 9: OOS returns vs CF VaR bands with breach highlights
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
fig.suptitle(
    "Fig 9 — OOS VaR Backtest (Jan 2022 – Jun 2026): ^NDX & ^GSPC Daily Returns vs. 99% CF VaR",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, res, color in zip(axes, [ndx_result, gspc_result],
                          [INDEX_COLORS["NDX"], INDEX_COLORS["GSPC"]]):
    oos_r   = res.oos_returns * 100   # convert to %
    var_neg = -res.is_var * 100       # lower VaR bound (negative = loss threshold)

    # All returns
    ax.bar(oos_r.index, oos_r.values,
           color=color, alpha=0.4, width=1.5, zorder=2, label="Daily log-return")

    # Breach days highlighted
    breach_r = oos_r[res.breach_mask]
    ax.bar(breach_r.index, breach_r.values,
           color="#B71C1C", alpha=0.9, width=1.5, zorder=4, label="VaR breach")

    # VaR boundary lines
    ax.axhline(var_neg, color="#B71C1C", linewidth=1.8, linestyle="--", zorder=5,
               label=f"−CF VaR 99% = {var_neg:.2f}%")
    ax.axhline(-var_neg, color="#2E7D32", linewidth=1.0, linestyle=":", alpha=0.6,
               label=f"+CF VaR 99% (upper) = {-var_neg:.2f}%")
    ax.axhline(0, color="#333333", linewidth=0.6, alpha=0.5)

    # Shade 2022 drawdown period
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.07, color="#B71C1C", label="2022 Drawdown window")

    # Annotate breach count
    ax.text(0.01, 0.04,
            f"Breaches: {res.oos_n_breaches}/{res.oos_n_obs} "
            f"({res.breach_rate*100:.2f}% vs 1.00% theoretical)\n"
            f"Kupiec p={res.kupiec['pvalue']:.4f}  "
            f"Christoffersen p={res.christoffersen['pvalue']:.4f}",
            transform=ax.transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.9))

    ax.set_title(f"{res.index_name} OOS (IS CF VaR = {res.is_var*100:.3f}%)", fontsize=11)
    ax.set_ylabel("Daily Log-Return (%)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.legend(fontsize=8.5, loc="upper right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

plt.tight_layout()
out9 = FIGURES_DIR / "fig9_var_backtest_oos.png"
plt.savefig(out9, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out9}")
plt.show()
