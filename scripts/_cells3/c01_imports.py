import sys, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/Vinesh/Documents/AIBubble")
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import set_paper_style, INDEX_COLORS, SCENARIO_COLORS
from src.models.var_pfe import calculate_cornish_fisher_var, calculate_historical_var
from src.models.validation import (
    TimeSeriesPartitioner,
    VaRBacktester,
    kupiec_pof_test,
    christoffersen_independence_test,
    christoffersen_combined_test,
    drawdown_velocity,
    IS_START, IS_END, OOS_START, OOS_END,
)

set_paper_style(dpi=150)

with (PROJECT_ROOT / "config" / "paths.yaml").open() as f:
    paths_cfg = yaml.safe_load(f)

CACHE_DIR   = PROJECT_ROOT / paths_cfg["data"]["cache"]
FIGURES_DIR = PROJECT_ROOT / paths_cfg["paper"]["figures"]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
print(f"IS window  : {IS_START} through {IS_END}")
print(f"OOS window : {OOS_START} through {OOS_END}")
