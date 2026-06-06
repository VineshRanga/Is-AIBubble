# Figure 10: Fold performance scatter matrix
n_folds    = len(ndx_result.fold_kupiec)
fold_nums  = np.arange(1, n_folds + 1)

ndx_kp_p   = [k["pvalue"]     for k in ndx_result.fold_kupiec]
gspc_kp_p  = [k["pvalue"]     for k in gspc_result.fold_kupiec]
ndx_br     = [k["breach_rate"]*100 for k in ndx_result.fold_kupiec]
gspc_br    = [k["breach_rate"]*100 for k in gspc_result.fold_kupiec]
ndx_lr     = [k["lr_stat"]    for k in ndx_result.fold_kupiec]
gspc_lr    = [k["lr_stat"]    for k in gspc_result.fold_kupiec]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Fig 10 — IS Cross-Validation Performance & Drawdown Velocity MAE\n"
    "(TimeSeriesSplit, 5 folds, 1995–2021)",
    fontsize=13, fontweight="bold", y=1.02
)

FOLD_COLORS = [INDEX_COLORS["NDX"], INDEX_COLORS["GSPC"]]
FOLD_LABELS = ["^NDX", "^GSPC"]

# Panel A: Kupiec p-value per fold
ax = axes[0, 0]
w  = 0.35
ax.bar(fold_nums - w/2, ndx_kp_p,  w, color=INDEX_COLORS["NDX"],  label="^NDX",  alpha=0.85)
ax.bar(fold_nums + w/2, gspc_kp_p, w, color=INDEX_COLORS["GSPC"], label="^GSPC", alpha=0.85)
ax.axhline(0.05, color="#B71C1C", linewidth=1.5, linestyle="--", label="p=0.05 threshold")
ax.set_title("Kupiec POF p-value per IS Fold", fontsize=11)
ax.set_xlabel("Fold", fontsize=10)
ax.set_ylabel("p-value", fontsize=10)
ax.set_xticks(fold_nums)
ax.legend(fontsize=9)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# Panel B: Breach rate vs theoretical 1%
ax = axes[0, 1]
ax.scatter(fold_nums, ndx_br,  color=INDEX_COLORS["NDX"],  s=80, zorder=4, label="^NDX",  marker="o")
ax.scatter(fold_nums, gspc_br, color=INDEX_COLORS["GSPC"], s=80, zorder=4, label="^GSPC", marker="s")
ax.axhline(1.0, color="#B71C1C", linewidth=1.5, linestyle="--", label="Theoretical 1%")
ax.set_title("Breach Rate per IS Fold (%)", fontsize=11)
ax.set_xlabel("Fold", fontsize=10)
ax.set_ylabel("Breach Rate (%)", fontsize=10)
ax.set_xticks(fold_nums)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# Panel C: Kupiec LR statistic per fold (vs chi-sq critical)
ax = axes[1, 0]
chi2_crit = stats_chi2_ppf = 3.841   # chi2(1) critical at 5%
from scipy.stats import chi2 as _chi2
chi2_crit = _chi2.ppf(0.95, df=1)
ax.bar(fold_nums - w/2, ndx_lr,  w, color=INDEX_COLORS["NDX"],  label="^NDX",  alpha=0.85)
ax.bar(fold_nums + w/2, gspc_lr, w, color=INDEX_COLORS["GSPC"], label="^GSPC", alpha=0.85)
ax.axhline(chi2_crit, color="#B71C1C", linewidth=1.5, linestyle="--",
           label=f"chi2 critical = {chi2_crit:.2f}")
ax.set_title("Kupiec LR Statistic per IS Fold", fontsize=11)
ax.set_xlabel("Fold", fontsize=10)
ax.set_ylabel("LR statistic", fontsize=10)
ax.set_xticks(fold_nums)
ax.legend(fontsize=9)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# Panel D: Drawdown velocity MAE
ax = axes[1, 1]
have_dd = bool(ndx_result.fold_dd_vel_mae and gspc_result.fold_dd_vel_mae)
if have_dd:
    n = min(len(ndx_result.fold_dd_vel_mae), len(gspc_result.fold_dd_vel_mae))
    fn = np.arange(1, n + 1)
    ax.bar(fn - w/2, ndx_result.fold_dd_vel_mae[:n],  w,
           color=INDEX_COLORS["NDX"],  label="^NDX",  alpha=0.85)
    ax.bar(fn + w/2, gspc_result.fold_dd_vel_mae[:n], w,
           color=INDEX_COLORS["GSPC"], label="^GSPC", alpha=0.85)
    ax.set_xticks(fn)
else:
    ax.text(0.5, 0.5, "Insufficient observations\nfor velocity MAE",
            transform=ax.transAxes, ha="center", va="center", fontsize=10)
ax.set_title("Drawdown Velocity MAE per Fold (%/day)", fontsize=11)
ax.set_xlabel("Fold", fontsize=10)
ax.set_ylabel("MAE (%/day)", fontsize=10)
ax.legend(fontsize=9)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

plt.tight_layout()
out10 = FIGURES_DIR / "fig10_drawdown_velocity_error.png"
plt.savefig(out10, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out10}")
plt.show()
