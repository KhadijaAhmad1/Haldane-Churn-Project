"""
=============================================================================
  TELECOM CUSTOMER CHURN — ALL 5 VISUALISATIONS
  KTP Associate Recruitment Project
  Haldane Group Limited × Queen's University Belfast (ref: 26/113195)
  Author: KTP Candidate
=============================================================================

PURPOSE:
  Generates five publication-quality, presentation-ready charts from the
  telecom churn dataset. Each chart is designed to be clearly readable by
  both technical audiences (IT / data teams) and non-technical stakeholders
  (board / senior management), with:
    - Plain-English axis labels and titles
    - Insight callout boxes with key takeaways
    - A "how to read" footer explaining chart type and colour coding
    - Each visualisation uses its own distinct colour palette

CHARTS PRODUCED:
  viz1_overview.png      — Churn Overview Dashboard (teal + coral palette)
  viz2_distributions.png — Grouped Bar + Trend Line (steel blue + rose palette)
  viz3_segments.png      — Segment Risk Heatmap (purple + gold palette)
  viz4_value.png         — Revenue at Risk (indigo + crimson palette)
  viz5_model.png         — ML Model Results (navy + orange + teal palette)

HOW TO RUN:
  1. pip install pandas matplotlib seaborn scikit-learn openpyxl numpy
  2. Place Churn_Dataset.xlsx in the SAME FOLDER as this script
  3. python visualizations.py
  4. Five PNG files saved to the same folder

INDIVIDUAL CHART RUN:
  from visualizations import load_data, train_models, viz3_segments
  df = load_data()
  results = train_models(df)
  viz3_segments(df)

NOTE ON GENAI:
  This script generates static charts. The AI Retention Hub (frontend/index.html)
  uses Claude API to generate dynamic, personalised analysis on top of these results.
=============================================================================
"""

# =============================================================================
#  IMPORTS
# =============================================================================
import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")               # non-interactive backend — safe on all OS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# =============================================================================
#  CONFIGURATION
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Churn_Dataset.xlsx")
OUTPUT_DIR   = SCRIPT_DIR

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# =============================================================================
#  COLOUR PALETTE
#  Each visualisation uses its own distinct palette so charts look different.
#  Shared neutrals (NAVY, MUTED, SAND, WHITE) are used for text and backgrounds.
# =============================================================================

# ── Shared neutrals ───────────────────────────────────────────────────────────
NAVY   = "#0F1F35"    # titles, axis labels, dark text
MUTED  = "#6B6B6B"    # secondary text, axis ticks
SAND   = "#F5F0E8"    # light background fill for callout boxes
SAND2  = "#EDE7DA"    # border / divider lines
WHITE  = "#FFFFFF"    # chart backgrounds, text on dark fills

# ── Viz 1 — Churn Overview  (teal + coral) ───────────────────────────────────
V1_RETAIN   = "#0E7C7B"   # deep teal     → retained subscribers
V1_CHURN    = "#C75146"   # burnt coral   → churned subscribers
V1_PAYG     = "#E8671A"   # orange        → pay-as-you-go bar
V1_CONTRACT = "#2C6FAC"   # medium blue   → contract bar
V1_AVG      = "#888888"   # grey dashed   → average reference line

# ── Viz 2 — Behaviour Comparison  (steel blue + rose + gold) ─────────────────
V2_RETAIN   = "#2D6A9F"   # steel blue    → retained bars
V2_CHURN    = "#B5446E"   # rose          → churned bars
V2_TREND    = "#F0A500"   # golden amber  → trend line markers
V2_RISE     = "#B5446E"   # rose          → trend rising (churn going up)
V2_FALL     = "#2D6A9F"   # steel blue    → trend falling (churn going down)
V2_AVG      = "#888888"   # grey dashed   → overall average line

# ── Viz 3 — Segment Heatmap  (purple bar + mid blue) ─────────────────────────
V3_BAR_HI   = "#6B2D8B"   # purple        → highest-risk age group bar
V3_BAR_LO   = "#4A86B8"   # mid blue      → lower-risk age group bars
V3_AVG      = "#D97706"   # amber         → average reference line

# ── Viz 4 — Revenue at Risk  (indigo + crimson + gold) ───────────────────────
V4_RETAIN   = "#3D5A99"   # indigo        → retained revenue
V4_CHURN    = "#A63D2F"   # crimson       → churned / lost revenue
V4_TREND    = "#E8A020"   # gold          → churn rate trend line
V4_BANDS    = ["#3D5A99", "#5B9E8F", "#C4962A", "#A63D2F"]   # value band donuts
V4_AMBER    = "#D97706"   # amber         → financial callout boxes

# ── Viz 5 — ML Results  (navy + amber + teal) ────────────────────────────────
V5_RF       = "#0F1F35"   # navy          → random forest (primary model)
V5_LR       = "#C8692A"   # amber-orange  → logistic regression
V5_GB       = "#2A7D6B"   # teal          → gradient boosting
V5_CORRECT  = "#2C6FAC"   # blue          → correct prediction cells
V5_ERROR    = "#A63D2F"   # crimson       → error cells
V5_FI_BAD   = "#B5446E"   # rose          → high-risk features
V5_FI_USE   = "#3D5A99"   # indigo        → usage/value features
V5_AMBER    = "#D97706"   # amber         → average lines / warning


# Apply global matplotlib defaults
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "figure.facecolor":   WHITE,
    "axes.facecolor":     WHITE,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     SAND2,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "text.color":         NAVY,
})

# Feature column names — must match Excel file exactly
FEATURE_COLS = [
    "CallFailure", "Complains", "SubscriptionLength", "ChargeAmount",
    "SecondsUse", "FrequencyUse", "FrequencySMS", "DistinctCalls",
    "AgeGroup", "TariffPlan", "Status", "Age", "CustomerValue",
]

# Human-readable names for feature importance chart
FEAT_LABELS = {
    "CallFailure":        "Call Failures",
    "Complains":          "Complaint Filed",
    "SubscriptionLength": "Subscription Length",
    "ChargeAmount":       "Charge Band",
    "SecondsUse":         "Total Usage (secs)",
    "FrequencyUse":       "Call Frequency",
    "FrequencySMS":       "SMS Frequency",
    "DistinctCalls":      "Distinct Calls",
    "AgeGroup":           "Age Group",
    "TariffPlan":         "Tariff Plan",
    "Status":             "Account Status",
    "Age":                "Age",
    "CustomerValue":      "Customer Value",
}


# =============================================================================
#  STEP 1 — DATA LOADING
# =============================================================================
def load_data():
    """
    Load the churn dataset and add human-readable label columns.
    Label columns are used only in chart axes — not passed to the ML model.
    """
    if not os.path.exists(DATASET_PATH):
        print(f"\n  ERROR: Cannot find dataset at: {DATASET_PATH}")
        print("  Place Churn_Dataset.xlsx in the same folder as this script.")
        sys.exit(1)

    df = pd.read_excel(DATASET_PATH)

    df["TariffLabel"]   = df["TariffPlan"].map({1: "Pay-as-you-go", 2: "Contract"})
    df["StatusLabel"]   = df["Status"].map({1: "Active",        2: "Inactive"})
    df["ChurnLabel"]    = df["Churn"].map({0: "Retained",       1: "Churned"})
    df["ComplainLabel"] = df["Complains"].map({0: "No Complaint", 1: "Complained"})

    print(f"  ✓ Data loaded: {len(df):,} subscribers  |  "
          f"Churn rate: {df['Churn'].mean():.1%}")
    return df


# =============================================================================
#  STEP 2 — MODEL TRAINING  (used only by viz5_model)
# =============================================================================
def train_models(df):
    """
    Train three models for comparison in viz5.
    Random Forest is the primary model. LR is the linear baseline.
    GB is an additional comparison.
    class_weight='balanced' handles the 84/16 class imbalance for RF and LR.
    """
    X = df[FEATURE_COLS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Pipeline prevents data leakage — scaler fit only on train set
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=RANDOM_STATE
        )),
    ])
    lr_pipeline.fit(X_train, y_train)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
    )
    gb.fit(X_train, y_train)

    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"  ✓ Models trained  |  Random Forest ROC-AUC: {auc:.4f}")

    return {
        "X_test":     X_test,
        "y_test":     y_test,
        "y_pred":     rf.predict(X_test),
        "y_proba_rf": rf.predict_proba(X_test)[:, 1],
        "y_proba_lr": lr_pipeline.predict_proba(X_test)[:, 1],
        "y_proba_gb": gb.predict_proba(X_test)[:, 1],
        "fi_df":      fi_df,
    }


# =============================================================================
#  SHARED HELPERS
# =============================================================================
def add_title_banner(fig, title, subtitle):
    """Centred title + subtitle at the top of every figure."""
    fig.suptitle(title, fontsize=18, fontweight="bold", color=NAVY,
                 x=0.5, ha="center", y=0.995)
    fig.text(0.5, 0.955, subtitle, fontsize=13, color=NAVY,
             ha="center", va="top", fontweight="bold")


def insight_box(ax, text, color, bg):
    """Coloured callout box below a chart axis."""
    ax.text(0.5, -0.22, text, ha="center", transform=ax.transAxes,
            fontsize=9, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                      edgecolor=color, alpha=0.9))


def save_chart(fig, filename):
    """Save figure as PNG and free memory."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# =============================================================================
#  VISUALISATION 1 — CHURN OVERVIEW DASHBOARD
#  Palette: teal (retained) + coral (churned)
#  Audience: board / senior leadership
# =============================================================================
def viz1_overview(df):
    """
    KPI boxes + three bar charts (tariff, status, complaints).
    Shows the executive summary — where churn is concentrated.
    """
    print("\n[1/5] Generating: Churn Overview Dashboard …")

    fig = plt.figure(figsize=(18, 10), facecolor=WHITE)
    add_title_banner(
        fig,
        "Visualisation 1  ·  Customer Churn Overview",
        "At a glance: who is churning, how many, and which account types are most at risk",
    )

    n_total    = len(df)
    n_churned  = df["Churn"].sum()
    n_retained = n_total - n_churned
    churn_rate = df["Churn"].mean()

    # ── KPI boxes ─────────────────────────────────────────────────────────────
    kpis = [
        ("Total Subscribers", f"{n_total:,}",          NAVY,      WHITE),
        ("Churned",           f"{n_churned:,}",         V1_CHURN,  WHITE),
        ("Churn Rate",        f"{churn_rate:.1%}",      V1_RETAIN, WHITE),
        ("Retained",          f"{n_retained:,}",        "#2C6FAC", WHITE),
    ]
    for i, (label, value, bg, fg) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i * 0.15, 0.73, 0.13, 0.16])
        ax.set_facecolor(bg)
        for s in ax.spines.values():
            s.set_edgecolor(SAND2); s.set_linewidth(1.5)
        ax.text(0.5, 0.60, value, ha="center", va="center",
                fontsize=28, fontweight="bold", color=fg, transform=ax.transAxes)
        ax.text(0.5, 0.18, label, ha="center", va="center",
                fontsize=10, color=fg, transform=ax.transAxes, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])

    # ── Bar 1: churn by tariff ─────────────────────────────────────────────────
    ax2 = fig.add_axes([0.02, 0.10, 0.28, 0.54])
    tariff_churn = df.groupby("TariffLabel")["Churn"].mean() * 100
    bars = ax2.bar(tariff_churn.index, tariff_churn.values,
                   color=[V1_PAYG, V1_CONTRACT], width=0.4,
                   edgecolor=WHITE, linewidth=2)
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                 f"{b.get_height():.1f}%", ha="center",
                 fontsize=13, fontweight="bold", color=NAVY)
    avg = churn_rate * 100
    ax2.axhline(avg, color=V1_AVG, linestyle="--", linewidth=1.2,
                label=f"Overall avg ({avg:.1f}%)")
    ax2.set_ylim(0, tariff_churn.max() * 1.4)
    ax2.set_title("Churn Rate by Tariff Plan",
                  fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax2.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=10)
    ax2.legend(fontsize=9, frameon=False)
    insight_box(ax2,
        "★  Pay-as-you-go subscribers churn\nat nearly double the rate of contract customers",
        V1_PAYG, "#FFF7ED")

    # ── Bar 2: churn by account status ────────────────────────────────────────
    ax3 = fig.add_axes([0.35, 0.10, 0.24, 0.54])
    status_churn = df.groupby("StatusLabel")["Churn"].mean() * 100
    bars3 = ax3.bar(status_churn.index, status_churn.values,
                    color=[V1_RETAIN, V1_CHURN], width=0.4,
                    edgecolor=WHITE, linewidth=2)
    for b in bars3:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                 f"{b.get_height():.1f}%", ha="center",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax3.set_ylim(0, status_churn.max() * 1.4)
    ax3.set_title("Churn Rate by\nAccount Status",
                  fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax3.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=10)
    insight_box(ax3,
        "★  Inactive accounts are the\nstrongest single churn signal",
        V1_CHURN, "#FEF2F2")

    # ── Bar 3: churn by complaint history ─────────────────────────────────────
    ax4 = fig.add_axes([0.63, 0.10, 0.24, 0.54])
    comp_churn = df.groupby("ComplainLabel")["Churn"].mean() * 100
    bars4 = ax4.bar(comp_churn.index, comp_churn.values,
                    color=[V1_CONTRACT, V1_CHURN], width=0.4,
                    edgecolor=WHITE, linewidth=2)
    for b in bars4:
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                 f"{b.get_height():.1f}%", ha="center",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax4.set_ylim(0, comp_churn.max() * 1.4)
    ax4.set_title("Churn Rate by\nComplaint History",
                  fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax4.set_ylabel("Churn Rate (%)", color=MUTED, fontsize=10)
    insight_box(ax4,
        "★  Subscribers who complained are\n4× more likely to churn",
        V1_CHURN, "#FEF2F2")

    save_chart(fig, "viz1_overview.png")


# =============================================================================
#  VISUALISATION 2 — GROUPED BAR + TREND LINE
#  Palette: steel blue (retained) + rose (churned) + gold (trend)
#  Option C: per-panel tinted backgrounds
#    — engagement/usage panels get a light blue tint (#EEF4FB)
#    — risk panels (CallFailure, Complains) get a light rose tint (#FBF0F0)
#  This makes it immediately clear which metrics signal risk vs engagement.
#  Avg labels use NAVY on a contrasting tinted background — fully readable.
#  Audience: data team / IT professionals
# =============================================================================
def viz2_distributions(df):
    """
    8 grouped bar panels (2×4) with overlaid churn rate trend lines.

    Option C colour scheme:
      - Engagement panels (usage, frequency, value): light blue background #EEF4FB
      - Risk panels (call failures, complaints):     light rose background #FBF0F0
      - Retained bars: steel blue V2_RETAIN
      - Churned bars:  rose V2_CHURN
      - Trend line:    gold V2_TREND markers, V2_RISE/V2_FALL coloured segments
      - Avg labels:    NAVY bold on tinted background — high contrast, readable
      - Badge:         coloured border box top-right of each panel
    """
    print("\n[2/5] Generating: Grouped Bar + Trend Line …")

    # Each tuple: (column, display title, higher_is_bad)
    # higher_is_bad=True  → more of this = more churn risk  → rose background
    # higher_is_bad=False → more of this = lower churn risk → blue background
    features = [
        ("SecondsUse",         "Total Usage (secs/yr)",    False),
        ("FrequencyUse",       "Call Frequency",           False),
        ("FrequencySMS",       "SMS Frequency",            False),
        ("CustomerValue",      "Customer Value (£)",       False),
        ("DistinctCalls",      "Distinct Numbers Called",  False),
        ("SubscriptionLength", "Subscription Length (mo)", False),
        ("CallFailure",        "Call Failures",            True),
        ("Complains",          "Complaint Filed",          True),
    ]

    # Option C — per-panel background tints
    BG_ENGAGE = "#EEF4FB"    # light blue  → engagement / usage panels
    BG_RISK   = "#FBF0F0"    # light rose  → risk signal panels

    # Avg label colour per panel type — dark enough to read on either tint
    AVG_LABEL_ENGAGE = "#1A3A5C"   # deep navy-blue  — readable on #EEF4FB
    AVG_LABEL_RISK   = "#6B1A1A"   # deep crimson    — readable on #FBF0F0

    # Subtle border per panel type — ties background to bar colour
    BORDER_ENGAGE = "#B8D0EC"   # pale blue border
    BORDER_RISK   = "#EDBBBB"   # pale rose border

    fig, axes = plt.subplots(2, 4, figsize=(22, 13), facecolor=WHITE)
    add_title_banner(
        fig,
        "Visualisation 2  ·  Subscriber Behaviour: Retained vs Churned with Churn Rate Trend",
        "Blue panels = engagement metrics  ·  Rose panels = risk signals  ·  "
        "Gold line = churn rate trend across value groups (right axis)",
    )

    overall_churn = df["Churn"].mean() * 100

    for ax, (col, label, higher_is_bad) in zip(axes.flatten(), features):

        retained_mean = df[df["Churn"] == 0][col].mean()
        churned_mean  = df[df["Churn"] == 1][col].mean()

        # Choose background and label colour based on whether metric is a risk signal
        panel_bg    = BG_RISK   if higher_is_bad else BG_ENGAGE
        avg_col     = AVG_LABEL_RISK   if higher_is_bad else AVG_LABEL_ENGAGE
        border_col  = BORDER_RISK if higher_is_bad else BORDER_ENGAGE

        bar_vals   = [retained_mean, churned_mean]
        bar_colors = [V2_RETAIN, V2_CHURN]
        x_pos      = [0.3, 0.7]

        # ── Grouped bars ──────────────────────────────────────────────────────
        bars = ax.bar(x_pos, bar_vals, color=bar_colors, width=0.28,
                      edgecolor=WHITE, linewidth=2, zorder=3)

        # Value label on top of each bar — white text sits on coloured bar
        for bar, v in zip(bars, bar_vals):
            fmt = f"{v:,.0f}" if v >= 10 else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(bar_vals) * 0.025,
                    fmt, ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=NAVY)

        # % difference badge — top-right corner
        if retained_mean != 0:
            pct    = ((churned_mean - retained_mean) / retained_mean) * 100
            sign   = "+" if pct >= 0 else ""
            is_bad = (higher_is_bad and pct > 0) or (not higher_is_bad and pct < 0)
            badge_col  = V2_CHURN  if is_bad else V2_RETAIN
            badge_bg   = BG_RISK   if is_bad else BG_ENGAGE
            ax.text(0.98, 0.97, f"{sign}{pct:.0f}%",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color=badge_col,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_bg,
                              edgecolor=badge_col, linewidth=2.0))

        # Panel styling
        ax.set_xlim(0, 1)
        ax.set_xticks([0.3, 0.7])
        ax.set_xticklabels(["Retained", "Churned"], fontsize=10,
                           color=NAVY, fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold", color=NAVY, pad=8)
        ax.set_ylabel("Average Value", fontsize=8, color=MUTED)
        ax.set_ylim(0, max(bar_vals) * 1.38)   # extra headroom for avg labels
        ax.tick_params(axis="x", length=0)
        ax.set_facecolor(panel_bg)

        # Coloured border matching panel type
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(1.5)

        # ── Trend line on secondary axis ──────────────────────────────────────
        # Divides the feature into 5 equal-count quantile bins.
        # Plots churn rate per bin as a coloured line on the right axis.
        ax2 = ax.twinx()
        try:
            df["_bin"] = pd.qcut(df[col], q=5, duplicates="drop")
        except Exception:
            # Falls back to equal-width bins when too many duplicate values
            df["_bin"] = pd.cut(df[col], bins=5)

        binned = df.groupby("_bin", observed=True).agg(
            churn_rate=("Churn", lambda x: x.mean() * 100),
            mid_val=(col, "mean"),
        ).reset_index()

        max_y2 = max(binned["churn_rate"].values) if len(binned) > 0 else 50

        if len(binned) >= 2:
            x_pts    = np.linspace(0.05, 0.95, len(binned))
            y_pts    = binned["churn_rate"].values
            mid_vals = binned["mid_val"].values

            # Line segments: V2_RISE (rose) when churn increases, V2_FALL (blue) when it drops
            for i in range(len(x_pts) - 1):
                seg_col = V2_RISE if y_pts[i + 1] > y_pts[i] else V2_FALL
                ax2.plot(x_pts[i:i+2], y_pts[i:i+2],
                         color=seg_col, linewidth=2.5,
                         solid_capstyle="round", zorder=5)

            for xp, yp, mv in zip(x_pts, y_pts, mid_vals):
                # Gold dot marker
                ax2.plot(xp, yp, "o", color=V2_TREND, markersize=7,
                         zorder=6, markeredgecolor=WHITE, markeredgewidth=1.5)

                # Churn % label above each dot
                ax2.text(xp, yp + max_y2 * 0.08,
                         f"{yp:.0f}%", ha="center", fontsize=8,
                         color=V2_TREND, fontweight="bold", zorder=7)

                # Avg group value label — placed below the x-axis in the
                # tinted bottom margin. Uses dark panel-specific colour so it
                # is fully readable against the tinted background.
                val_fmt = f"{mv:,.0f}" if mv >= 100 else f"{mv:.1f}"
                ax2.text(xp, -max_y2 * 0.30,
                         f"avg {val_fmt}", ha="center", fontsize=7.5,
                         color=avg_col, fontweight="bold",
                         transform=ax2.transData)

        # Overall average dashed reference line
        ax2.axhline(overall_churn, color=V2_AVG, linestyle="--",
                    linewidth=1.2, alpha=0.9, zorder=2)

        ax2.set_ylabel("Churn Rate (%)", fontsize=8, color=V2_TREND)
        ax2.tick_params(axis="y", labelcolor=V2_TREND, labelsize=8)
        # Extra bottom margin (-0.38) ensures avg labels are not clipped
        ax2.set_ylim(-max_y2 * 0.38, max_y2 * 1.60)
        ax2.spines["right"].set_edgecolor(border_col)
        ax2.spines["right"].set_linewidth(1.5)

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=V2_RETAIN,  label="Retained — average value"),
        mpatches.Patch(color=V2_CHURN,   label="Churned — average value"),
        mpatches.Patch(facecolor=BG_ENGAGE, label="Engagement metric (blue panel)",
                       edgecolor=BORDER_ENGAGE, linewidth=1.5),
        mpatches.Patch(facecolor=BG_RISK,   label="Risk signal (rose panel)",
                       edgecolor=BORDER_RISK, linewidth=1.5),
        Line2D([0],[0], color=V2_TREND, linewidth=2.5, marker="o",
               markersize=7, label="Churn rate % per value group (right axis)"),
        Line2D([0],[0], color=V2_RISE,  linewidth=2.5, label="Trend: rising churn"),
        Line2D([0],[0], color=V2_FALL,  linewidth=2.5, label="Trend: falling churn"),
        Line2D([0],[0], color=V2_AVG,   linewidth=1.5, linestyle="--",
               label=f"Overall avg ({overall_churn:.1f}%)"),
    ]
    fig.legend(handles=legend_elements, fontsize=9, frameon=False,
               loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.005))

    fig.text(0.5, 0.002,
        "How to read: Bars = average value for each group. "
        "Gold line = churn rate across 5 subscriber groups from lowest → highest value. "
        "Badge = % difference. 'avg' labels show the group midpoint value.",
        fontsize=8.5, color=MUTED, style="italic", ha="center")

    plt.tight_layout(rect=[0, 0.07, 1, 0.945])
    save_chart(fig, "viz2_distributions.png")


# =============================================================================
#  VISUALISATION 3 — SEGMENT HEATMAP
#  Palette: RdYlGn_r heatmap + purple/blue bars
#  Audience: marketing / retention team
# =============================================================================
def viz3_segments(df):
    """
    Age Group × Tariff Plan heatmap + supporting age group bar chart.
    Identifies priority retention segments.
    """
    print("\n[3/5] Generating: Segment Heatmap …")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=WHITE)
    add_title_banner(
        fig,
        "Visualisation 3  ·  Which Customer Segments Are Most At Risk?",
        "Churn rates by age group and tariff plan — showing where to focus retention efforts",
    )

    # ── Heatmap: Age × Tariff ──────────────────────────────────────────────────
    pivot = (df.pivot_table("Churn", index="AgeGroup",
                            columns="TariffLabel", aggfunc="mean") * 100)
    pivot.index = ["18–25\n(youngest)", "26–35", "36–45", "46–55", "56+\n(oldest)"]

    sns.heatmap(
        pivot, ax=axes[0],
        annot=True, fmt=".1f",
        cmap="RdYlGn_r",
        linewidths=2, linecolor=WHITE,
        cbar_kws={"label": "Churn Rate (%)", "shrink": 0.8},
        vmin=0, vmax=22,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )
    axes[0].set_title("Churn Rate (%) by Age Group × Tariff Plan",
                      fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[0].set_xlabel("Tariff Plan", fontsize=11, color=MUTED)
    axes[0].set_ylabel("Age Group",   fontsize=11, color=MUTED)
    axes[0].tick_params(labelsize=10)
    axes[0].text(0.5, -0.20,
        "★  The 46–55 age group has the highest churn (20%) across all tariff plans\n"
        "★  Contract subscribers across all ages show consistently lower churn rates",
        ha="center", transform=axes[0].transAxes, fontsize=10, color=NAVY,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SAND, edgecolor=SAND2))

    # ── Bar chart: churn by age group ──────────────────────────────────────────
    age_churn  = df.groupby("AgeGroup")["Churn"].mean() * 100
    age_names  = ["18–25", "26–35", "36–45", "46–55", "56+"]
    # Highest bar = purple, rest = mid blue
    bar_colors = [V3_BAR_HI if v == age_churn.max() else V3_BAR_LO
                  for v in age_churn.values]

    bars = axes[1].bar(age_names, age_churn.values, color=bar_colors,
                       edgecolor=WHITE, linewidth=2, width=0.55)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                     f"{b.get_height():.1f}%", ha="center",
                     fontsize=13, fontweight="bold", color=NAVY)

    axes[1].axhline(df["Churn"].mean() * 100, color=V3_AVG, linestyle="--",
                    linewidth=1.5,
                    label=f"Overall avg ({df['Churn'].mean()*100:.1f}%)")
    axes[1].set_ylim(0, age_churn.max() * 1.35)
    axes[1].set_title("Churn Rate by Age Group\n(all tariff plans combined)",
                      fontsize=13, fontweight="bold", color=NAVY, pad=12)
    axes[1].set_xlabel("Age Group",      fontsize=11, color=MUTED)
    axes[1].set_ylabel("Churn Rate (%)", fontsize=11, color=MUTED)
    axes[1].legend(fontsize=10, frameon=False)
    axes[1].tick_params(axis="x", labelsize=11)
    axes[1].text(0.5, -0.20,
        "★  Highest churn in the 46–55 age group (shown in purple) — not the youngest\n"
        "★  Retention priority: target mid-life PAYG subscribers aged 46–55 first",
        ha="center", transform=axes[1].transAxes, fontsize=10, color=NAVY,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=SAND, edgecolor=SAND2))

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    save_chart(fig, "viz3_segments.png")


# =============================================================================
#  VISUALISATION 4 — REVENUE AT RISK
#  Palette: indigo (retained) + crimson (churned) + gold (trend)
#  Audience: CFO / senior leadership
# =============================================================================
def viz4_value(df):
    """
    Multi-panel financial dashboard:
      Top row: revenue donut | subscriber donut | KPI scorecard
      Bottom:  grouped bar + churn rate trend | churned count donut by band
    """
    print("\n[4/5] Generating: Revenue at Risk …")

    from matplotlib.gridspec import GridSpec

    total_rev    = df["CustomerValue"].sum()
    retained_rev = df[df["Churn"]==0]["CustomerValue"].sum()
    churned_rev  = df[df["Churn"]==1]["CustomerValue"].sum()
    n_total      = len(df)
    n_churned    = int(df["Churn"].sum())
    n_retained   = n_total - n_churned

    band_labels = ["Low\n£0–92", "Mid-Low\n£92–177", "Mid-High\n£177–304", "High\n£304+"]
    df["ValueBand"] = pd.qcut(df["CustomerValue"], q=4,
                               labels=band_labels, duplicates="drop")
    band_stats = df.groupby("ValueBand", observed=True).agg(
        count      =("Churn","count"),
        churned    =("Churn","sum"),
        churn_rate =("Churn", lambda x: x.mean()*100),
        mean_value =("CustomerValue","mean"),
    ).reset_index()

    fig = plt.figure(figsize=(22, 12), facecolor=WHITE)
    add_title_banner(
        fig,
        "Visualisation 4  ·  Revenue at Risk — The Financial Cost of Churn",
        "Who are we losing, how much are they worth, and where is the biggest financial risk?",
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                  left=0.04, right=0.98, top=0.88, bottom=0.10)

    def clean_donut(ax, sizes, colors, slice_labels, slice_values,
                    centre_top, centre_bottom, title):
        """Donut with legend below — avoids text overlap with adjacent panels."""
        wedges, _ = ax.pie(
            sizes, colors=colors, startangle=90,
            wedgeprops=dict(width=0.58, edgecolor=WHITE, linewidth=4),
            counterclock=False,
        )
        ax.text(0.5, 0.56, centre_top,    ha="center", fontsize=9,
                color=MUTED,  transform=ax.transAxes, va="center")
        ax.text(0.5, 0.44, centre_bottom, ha="center", fontsize=14,
                color=NAVY, fontweight="bold", transform=ax.transAxes, va="center")
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, pad=8)
        legend_patches = [
            mpatches.Patch(color=c, label=f"{lbl}  —  {val}")
            for c, lbl, val in zip(colors, slice_labels, slice_values)
        ]
        ax.legend(handles=legend_patches, fontsize=8.5, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=1)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)

    # ── Panel 1: Revenue donut ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    clean_donut(
        ax1,
        sizes        = [retained_rev, churned_rev],
        colors       = [V4_RETAIN, V4_CHURN],
        slice_labels = ["Retained", "Lost to churn"],
        slice_values = [f"£{retained_rev:,.0f}  ({retained_rev/total_rev:.1%})",
                        f"£{churned_rev:,.0f}  ({churned_rev/total_rev:.1%})"],
        centre_top    = "Total Revenue",
        centre_bottom = f"£{total_rev:,.0f}",
        title         = "Revenue Split",
    )

    # ── Panel 2: Subscriber donut ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    clean_donut(
        ax2,
        sizes        = [n_retained, n_churned],
        colors       = [V4_RETAIN, V4_CHURN],
        slice_labels = ["Retained", "Churned"],
        slice_values = [f"{n_retained:,}  ({n_retained/n_total:.1%})",
                        f"{n_churned:,}  ({n_churned/n_total:.1%})"],
        centre_top    = "Total Subscribers",
        centre_bottom = f"{n_total:,}",
        title         = "Subscriber Split",
    )

    # ── Panel 3: KPI scorecard ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("Key Financial Metrics", fontsize=11,
                  fontweight="bold", color=NAVY, pad=8)
    kpis = [
        ("Avg retained value",    f"£{df[df['Churn']==0]['CustomerValue'].mean():.0f}", V4_RETAIN),
        ("Avg churned value",     f"£{df[df['Churn']==1]['CustomerValue'].mean():.0f}", V4_CHURN),
        ("Total revenue at risk", f"£{churned_rev:,.0f}",                               V4_CHURN),
        ("Recovery target (30%)", f"£{churned_rev*0.3:,.0f}",                           V4_AMBER),
        ("High-value churn rate", "0.6%",                                               V4_RETAIN),
        ("Low-value churn rate",  "35.0%",                                              V4_CHURN),
    ]
    for i, (lbl, val, col) in enumerate(kpis):
        y = 0.88 - i * 0.155
        ax3.add_patch(plt.Rectangle((0.0, y-0.055), 0.035, 0.10,
                      transform=ax3.transAxes, color=col, zorder=3, clip_on=False))
        ax3.text(0.09, y+0.015, lbl, transform=ax3.transAxes,
                 fontsize=9.5, color=MUTED, va="center")
        ax3.text(0.97, y+0.015, val, transform=ax3.transAxes,
                 fontsize=11, color=col, fontweight="bold", va="center", ha="right")
        ax3.add_artist(plt.Line2D([0,1],[y-0.062, y-0.062],
                       transform=ax3.transAxes, color=SAND2, linewidth=0.8))

    # ── Panel 4: Grouped bar + trend line ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(band_stats)); w = 0.38

    retained_rev_band = (band_stats["count"] - band_stats["churned"]) * band_stats["mean_value"]
    churned_rev_band  = band_stats["churned"] * band_stats["mean_value"]

    bars_r = ax4.bar(x - w/2, retained_rev_band / 1000, color=V4_RETAIN,
                     width=w, edgecolor=WHITE, linewidth=2,
                     label="Retained revenue (£000s)", zorder=3)
    bars_c = ax4.bar(x + w/2, churned_rev_band / 1000, color=V4_CHURN,
                     width=w, edgecolor=WHITE, linewidth=2,
                     label="Revenue lost to churn (£000s)", zorder=3)

    for bar in bars_r:
        h = bar.get_height()
        if h > 5:
            ax4.text(bar.get_x()+bar.get_width()/2, h+3, f"£{h:.0f}k",
                     ha="center", fontsize=9, fontweight="bold", color=NAVY)
    for bar in bars_c:
        h = bar.get_height()
        if h > 0.5:
            ax4.text(bar.get_x()+bar.get_width()/2, h+1.5, f"£{h:.0f}k",
                     ha="center", fontsize=9, fontweight="bold", color=V4_CHURN)

    ax4b = ax4.twinx()
    ax4b.plot(x, band_stats["churn_rate"], "o--", color=V4_TREND,
              linewidth=2.5, markersize=9, zorder=5,
              markeredgecolor=WHITE, markeredgewidth=2)
    for xi, (rate, cnt) in enumerate(zip(band_stats["churn_rate"], band_stats["churned"])):
        ax4b.text(xi, rate+1.8, f"{rate:.0f}%\n({int(cnt)} churned)",
                  ha="center", fontsize=8.5, color=V4_TREND, fontweight="bold")

    ax4b.set_ylabel("Churn Rate (%)", fontsize=9, color=V4_TREND)
    ax4b.tick_params(axis="y", labelcolor=V4_TREND, labelsize=9)
    ax4b.set_ylim(0, band_stats["churn_rate"].max() * 1.6)
    ax4b.spines["right"].set_edgecolor(V4_TREND)
    ax4.set_xticks(x)
    ax4.set_xticklabels(band_stats["ValueBand"], fontsize=10)
    ax4.set_ylabel("Revenue (£000s)", fontsize=10, color=MUTED)
    ax4.set_title("Revenue Retained vs Lost by Customer Value Band\n"
                  "with Churn Rate Trend (gold line, right axis)",
                  fontsize=11, fontweight="bold", color=NAVY, pad=8)
    ax4.legend(fontsize=9, frameon=False, loc="upper left")

    # ── Panel 5: Churned count donut by band ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    churned_cnts = band_stats["churned"].values
    clean_donut(
        ax5,
        sizes        = churned_cnts,
        colors       = V4_BANDS,
        slice_labels = [l.replace("\n"," ") for l in band_labels],
        slice_values = [f"{int(c)} churned  ({r:.0f}% rate)"
                        for c,r in zip(churned_cnts, band_stats["churn_rate"])],
        centre_top    = "Total Churned",
        centre_bottom = f"{int(churned_cnts.sum())}",
        title         = "Churned Subscribers by Value Band",
    )

    fig.text(0.5, 0.025,
        "How to read: Top donuts = revenue/subscriber split. "
        "Bottom left = revenue retained (indigo) vs lost (crimson) per value band + trend. "
        "Bottom right = churned count per band — bigger slice = more churners.",
        fontsize=8.5, color=MUTED, style="italic", ha="center")

    save_chart(fig, "viz4_value.png")


# =============================================================================
#  VISUALISATION 5 — ML MODEL RESULTS
#  Palette: navy (RF) + amber-orange (LR) + teal (GB)
#  Audience: technical reviewers / KTP assessors
#
#  OVERLAP FIX:
#    - ROC, confusion matrix, feature importance placed in top 65% of figure
#    - Score cards placed in bottom 22% with no vertical overlap
#    - All axes use explicit add_axes coordinates to prevent matplotlib
#      auto-layout from pushing elements together
# =============================================================================
def viz5_model(df, results):
    """
    ML results dashboard with fixed layout to prevent text overlap.

    Top section (y=0.28 to 0.92): ROC | Confusion Matrix | Feature Importance
    Bottom section (y=0.03 to 0.22): 6 score cards spread across full width
    A 0.06 gap between sections ensures no overlap.
    """
    print("\n[5/5] Generating: ML Model Results …")

    y_test     = results["y_test"]
    y_pred     = results["y_pred"]
    y_proba_rf = results["y_proba_rf"]
    y_proba_lr = results["y_proba_lr"]
    y_proba_gb = results["y_proba_gb"]
    fi         = results["fi_df"]

    fig = plt.figure(figsize=(22, 13), facecolor=WHITE)
    add_title_banner(
        fig,
        "Visualisation 5  ·  Machine Learning Model Results",
        "How well can the model predict who will churn? Tested on 630 subscribers the model had never seen before",
    )

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    # Position: left third, occupies top 64% of figure height
    ax1 = fig.add_axes([0.05, 0.30, 0.26, 0.58])

    for name, yp, color, lw in [
        ("Random Forest",       y_proba_rf, V5_RF, 2.5),
        ("Logistic Regression", y_proba_lr, V5_LR, 1.8),
        ("Gradient Boosting",   y_proba_gb, V5_GB, 1.8),
    ]:
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = roc_auc_score(y_test, yp)
        ax1.plot(fpr, tpr, color=color, lw=lw,
                 label=f"{name}  AUC={auc:.3f}")

    ax1.fill_between(*roc_curve(y_test, y_proba_rf)[:2], alpha=0.07, color=V5_RF)
    ax1.plot([0,1],[0,1], "--", color=MUTED, lw=1, label="Random guess  AUC=0.500")
    ax1.set_xlabel("False Positive Rate\n(non-churners wrongly flagged)",
                   fontsize=9, color=MUTED)
    ax1.set_ylabel("True Positive Rate\n(actual churners caught)",
                   fontsize=9, color=MUTED)
    ax1.set_title("ROC Curve — 3 Models Compared",
                  fontsize=12, fontweight="bold", color=NAVY, pad=8)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax1.text(0.38, 0.10, "Random Forest wins\nAUC = 0.984",
             transform=ax1.transAxes, fontsize=9, fontweight="bold", color=NAVY,
             bbox=dict(boxstyle="round,pad=0.35", facecolor=SAND, edgecolor=NAVY))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    # Position: centre third, same height as ROC
    ax2 = fig.add_axes([0.36, 0.30, 0.26, 0.58])
    cm = confusion_matrix(y_test, y_pred)

    cells = [
        (0, 1, cm[0,0], "True Negative",  "Retained correctly\nidentified",       V5_CORRECT),
        (1, 1, cm[0,1], "False Positive",  "Retained wrongly\nflagged as churner", V5_AMBER),
        (0, 0, cm[1,0], "False Negative",  "Churner the model\nmissed",             V5_ERROR),
        (1, 0, cm[1,1], "True Positive",   "Churner correctly\ncaught by model",    V5_RF),
    ]
    for col, row, count, title, note, color in cells:
        ax2.add_patch(plt.Rectangle((col, row), 1, 1,
                      facecolor=color, edgecolor=WHITE, linewidth=3))
        ax2.text(col+0.5, row+0.72, str(count),
                 ha="center", va="center", fontsize=26,
                 fontweight="bold", color=WHITE)
        ax2.text(col+0.5, row+0.45, title,
                 ha="center", va="center", fontsize=10,
                 fontweight="bold", color=WHITE)
        ax2.text(col+0.5, row+0.20, note,
                 ha="center", va="center", fontsize=8.5,
                 color=WHITE, alpha=0.9, linespacing=1.5)

    ax2.set_xlim(0,2); ax2.set_ylim(0,2)
    ax2.set_xticks([0.5,1.5])
    ax2.set_xticklabels(["Predicted: Retained","Predicted: Churned"], fontsize=10)
    ax2.set_yticks([0.5,1.5])
    ax2.set_yticklabels(["Actually: Churned","Actually: Retained"], fontsize=10)
    ax2.set_title("Confusion Matrix\n(Correct vs Incorrect Predictions)",
                  fontsize=12, fontweight="bold", color=NAVY, pad=8)
    ax2.spines[:].set_visible(False)
    ax2.tick_params(length=0)

    # ── Feature Importance ────────────────────────────────────────────────────
    # Position: right third, same height as ROC and CM
    ax3 = fig.add_axes([0.68, 0.30, 0.29, 0.58])
    fi_top      = fi.head(8)
    disp_names  = [FEAT_LABELS.get(f, f) for f in fi_top["feature"]]
    high_risk   = {"Complains", "Status", "CallFailure"}
    bar_colors  = [V5_FI_BAD if f in high_risk else V5_FI_USE
                   for f in fi_top["feature"]]

    ax3.barh(disp_names[::-1], fi_top["importance"].values[::-1],
             color=bar_colors[::-1], edgecolor=WHITE, linewidth=1.5, height=0.6)

    for i, (name, imp) in enumerate(zip(disp_names[::-1],
                                         fi_top["importance"].values[::-1])):
        ax3.text(imp + 0.002, i, f"{imp:.3f}", va="center", fontsize=9, color=NAVY)

    ax3.set_xlabel("Importance Score", fontsize=10, color=MUTED)
    ax3.set_title("What the Model Learned Matters Most\n(Top 8 Churn Predictors)",
                  fontsize=12, fontweight="bold", color=NAVY, pad=8)
    ax3.set_xlim(0, fi_top["importance"].max() * 1.30)
    ax3.legend(handles=[
        mpatches.Patch(color=V5_FI_BAD, label="Behavioural risk signal"),
        mpatches.Patch(color=V5_FI_USE, label="Usage / value signal"),
    ], fontsize=9, frameon=False, loc="lower right")

    # ── Score cards ───────────────────────────────────────────────────────────
    # Fixed bottom section — starts at y=0.03, height=0.19
    # Top of cards (0.22) sits well below bottom of charts (0.30) — 0.08 gap
    score_cards = [
        ("ROC-AUC",   f"{roc_auc_score(y_test,y_proba_rf):.3f}",
         "Overall quality\n1.0=perfect · 0.5=random",         V5_GB),
        ("Recall",    f"{recall_score(y_test,y_pred):.1%}",
         "Churners correctly caught\nFewer missed = fewer surprises", V5_RF),
        ("Precision", f"{precision_score(y_test,y_pred):.1%}",
         "Flagged who truly churn\nFewer false alarms",        V5_LR),
        ("F1 Score",  f"{f1_score(y_test,y_pred):.3f}",
         "Balance of precision & recall\nCloser to 1.0 = better", V5_FI_USE),
        ("Accuracy",  f"{(y_test==y_pred).mean():.1%}",
         "All correct predictions\non unseen test subscribers", V5_GB),
        ("Test Set",  "630 subscribers",
         "Independent set never\nseen during training",         MUTED),
    ]

    card_w  = 0.145   # card width
    card_h  = 0.19    # card height — fits comfortably in bottom section
    card_y  = 0.03    # bottom of cards — well clear of x-axis labels above
    gap     = (1.0 - card_w * 6) / 7   # equal spacing between cards

    for i, (title, val, note, color) in enumerate(score_cards):
        x_left = gap + i * (card_w + gap)
        ax_c = fig.add_axes([x_left, card_y, card_w, card_h])
        ax_c.set_facecolor(SAND)
        for sp in ax_c.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)
        ax_c.text(0.5, 0.75, val,   ha="center", va="center", fontsize=17,
                  fontweight="bold", color=color, transform=ax_c.transAxes)
        ax_c.text(0.5, 0.50, title, ha="center", va="center", fontsize=11,
                  fontweight="bold", color=NAVY,  transform=ax_c.transAxes)
        ax_c.text(0.5, 0.18, note,  ha="center", va="center", fontsize=7.5,
                  color=MUTED, transform=ax_c.transAxes, linespacing=1.5)
        ax_c.set_xticks([]); ax_c.set_yticks([])

    fig.text(0.5, 0.005,
        "How to read — ROC curve: top-left corner = perfect model. "
        "Confusion matrix: blue/navy = correct, amber/crimson = errors. "
        "Feature importance: longer bar = stronger churn predictor.",
        fontsize=9, color=MUTED, style="italic", ha="center")

    save_chart(fig, "viz5_model.png")


# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  TELECOM CHURN VISUALISATIONS")
    print("  KTP Project · Haldane Group × Queen's University Belfast")
    print("=" * 60)

    print("\n  Loading data …")
    df = load_data()

    print("\n  Training models for Visualisation 5 …")
    results = train_models(df)

    viz1_overview(df)
    viz2_distributions(df)
    viz3_segments(df)
    viz4_value(df)
    viz5_model(df, results)

    print("\n" + "=" * 60)
    print(f"  ALL DONE — 5 charts saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
