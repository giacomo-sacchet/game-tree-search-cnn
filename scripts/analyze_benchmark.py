# =============================================================================
# analyze_benchmark.py
# Post-hoc statistical analysis of the Othello benchmark report written by
# 6_Othello.py (CompleteTest -> othello_benchmark_results.txt).
#
# It reads only the per-game logs ("INDIVIDUAL GAME LOG" blocks), so it needs
# no re-run of any game, and produces:
#
#   1. DRAW SENSITIVITY
#      Player 1 win rate and 95% Wilson CI under two conventions:
#        (a) draws counted as non-wins:  W / n            (thesis convention)
#        (b) draws excluded:             W / (W + L)      (decisive games only)
#      -> draws_sensitivity.csv
#
#   2. COLOUR (FIRST-MOVE) EFFECTS
#      Model 1 - per phase: 2x2 table (P1 colour x P1 won / did not win),
#                Fisher exact test, odds ratio, Holm-adjusted p-values.
#      Model 2 - across phases: Cochran-Mantel-Haenszel test of a common
#                colour effect, pooled odds ratio, and Breslow-Day test of
#                homogeneity of the odds ratios across phases.
#      Also reported: Black's win rate in each phase, whoever plays Black.
#      -> colour_effects.csv, fig_colour_effects.png
#
#   3. FULL RESULTS (thesis Table tab:full_results and Figure winrates.png)
#      Per phase: N, P1 win / draw / P2 win %, 95% Wilson CI on P1 win rate
#      (draws = non-wins), average seconds per game.
#      -> full_results.csv, full_results_rows.tex, winrates.png
#
# Usage:
#   python analyze_benchmark.py                       (report in current folder)
#   python analyze_benchmark.py --report path/othello_benchmark_results.txt \
#                               --out results/benchmark_analysis
#
# Requirements: numpy, pandas, scipy, statsmodels, matplotlib
# =============================================================================

import argparse
import math
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(ROOT, "results", "benchmark")
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Parsing
# ---------------------------------------------------------------------------

_PHASE_RE = re.compile(r"^\s*PHASE:\s*(.+?)\s*$")
_GAME_RE  = re.compile(r"^\s*(\d+),\s*(Black|White),\s*(\d),\s*([01]),\s*([\d.]+)\s*$")


def parse_report(path):
    """
    Return a DataFrame with one row per game:
      phase, game_id, p1_colour, winner_code, outcome
    where outcome is Player 1's result: 'W', 'D' or 'L'.
    """
    rows, phase = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _PHASE_RE.match(line)
            if m:
                phase = m.group(1)
                continue
            m = _GAME_RE.match(line)
            if m and phase is not None:
                gid, colour, code, p1_won, dur = m.groups()
                code = int(code)
                if int(p1_won) == 1:
                    outcome = "W"
                elif code == 4:
                    outcome = "D"
                else:
                    outcome = "L"
                rows.append((phase, int(gid), colour, code, outcome, float(dur)))
    df = pd.DataFrame(rows, columns=["phase", "game_id", "p1_colour",
                                     "winner_code", "outcome", "duration_s"])
    if df.empty:
        raise RuntimeError(f"No per-game records found in {path}")
    return df


def short_label(phase):
    """'9.  Minimax AB CNN vs Minimax AB' -> ('9', 'Minimax AB CNN vs Minimax AB')."""
    num, _, rest = phase.partition(".")
    return num.strip(), rest.strip()


# ---------------------------------------------------------------------------
# 2. Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Wilson score interval (same formula as _wilson_ci in 6_Othello.py)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - spread), min(1.0, centre + spread)


def verdict(lo, hi):
    if lo > 0.5:
        return "P1 better"
    if hi < 0.5:
        return "P2 better"
    return "n.s."


def counts(sub):
    c = sub["outcome"].value_counts()
    return int(c.get("W", 0)), int(c.get("D", 0)), int(c.get("L", 0))


# ---------------------------------------------------------------------------
# 3. Draw sensitivity
# ---------------------------------------------------------------------------

def draw_sensitivity(df):
    out = []
    for phase, sub in df.groupby("phase", sort=False):
        W, D, L = counts(sub)
        n = W + D + L
        p_a, lo_a, hi_a = wilson_ci(W, n)          # (a) draws = non-wins
        p_b, lo_b, hi_b = wilson_ci(W, W + L)      # (b) decisive games only
        out.append({
            "phase": phase, "n": n, "W": W, "D": D, "L": L,
            "rate_all": p_a, "ci_all_lo": lo_a, "ci_all_hi": hi_a,
            "verdict_all": verdict(lo_a, hi_a),
            "n_decisive": W + L,
            "rate_dec": p_b, "ci_dec_lo": lo_b, "ci_dec_hi": hi_b,
            "verdict_dec": verdict(lo_b, hi_b),
        })
    res = pd.DataFrame(out)
    res["verdict_changed"] = res["verdict_all"] != res["verdict_dec"]
    return res


# ---------------------------------------------------------------------------
# 4. Colour effects
# ---------------------------------------------------------------------------

def colour_effects(df):
    """
    Model 1: per-phase Fisher exact test on the 2x2 table
                         P1 won   P1 did not win
        P1 = Black         a            b
        P1 = White         c            d
    OR > 1 : Player 1 does better when moving first (Black).
    """
    rows, tables = [], []
    for phase, sub in df.groupby("phase", sort=False):
        Wb, Db, Lb = counts(sub[sub["p1_colour"] == "Black"])
        Ww, Dw, Lw = counts(sub[sub["p1_colour"] == "White"])
        nb, nw = Wb + Db + Lb, Ww + Dw + Lw
        table = np.array([[Wb, nb - Wb], [Ww, nw - Ww]])
        tables.append(table)
        odds, p = fisher_exact(table)
        _, lo_b, hi_b = wilson_ci(Wb, nb)
        _, lo_w, hi_w = wilson_ci(Ww, nw)
        # Black's win rate whoever plays Black: P1 wins in block A + P2 wins in block B
        black_wins = Wb + Lw
        _, lo_k, hi_k = wilson_ci(black_wins, nb + nw)
        rows.append({
            "phase": phase,
            "n_black": nb, "W_black": Wb, "D_black": Db, "L_black": Lb,
            "rate_black": Wb / nb, "ci_black_lo": lo_b, "ci_black_hi": hi_b,
            "n_white": nw, "W_white": Ww, "D_white": Dw, "L_white": Lw,
            "rate_white": Ww / nw, "ci_white_lo": lo_w, "ci_white_hi": hi_w,
            "odds_ratio": odds, "fisher_p": p,
            "black_side_rate": black_wins / (nb + nw),
            "black_side_lo": lo_k, "black_side_hi": hi_k,
        })
    res = pd.DataFrame(rows)
    # Holm correction for the 10 simultaneous per-phase tests
    res["fisher_p_holm"] = multipletests(res["fisher_p"], method="holm")[1]
    res["significant_holm"] = res["fisher_p_holm"] < 0.05

    # Model 2: Cochran-Mantel-Haenszel across phases + Breslow-Day homogeneity.
    # shift_zeros adds 0.5 to tables with a zero cell (phases at 100% win rate).
    st = StratifiedTable(tables, shift_zeros=True)
    cmh = st.test_null_odds(correction=True)
    bd  = st.test_equal_odds(adjust=True)          # Breslow-Day with Tarone adj.
    lo, hi = st.oddsratio_pooled_confint()
    pooled = {
        "pooled_OR": st.oddsratio_pooled, "pooled_OR_lo": lo, "pooled_OR_hi": hi,
        "cmh_stat": cmh.statistic, "cmh_p": cmh.pvalue,
        "bd_stat": bd.statistic, "bd_p": bd.pvalue,
    }

    # Sensitivity: the same two tests without the phases flagged by Model 1
    keep = [t for t, s in zip(tables, res["significant_holm"]) if not s]
    if 2 <= len(keep) < len(tables):
        st2  = StratifiedTable(keep, shift_zeros=True)
        cmh2 = st2.test_null_odds(correction=True)
        bd2  = st2.test_equal_odds(adjust=True)
        pooled.update({
            "excl_n_phases": len(keep),
            "excl_pooled_OR": st2.oddsratio_pooled,
            "excl_cmh_p": cmh2.pvalue, "excl_bd_p": bd2.pvalue,
        })
    return res, pooled


# ---------------------------------------------------------------------------
# 5. Figure
# ---------------------------------------------------------------------------

C_BLACK = "#2a78d6"     # blue   - P1 plays Black (moves first)
C_WHITE = "#eb6834"     # orange - P1 plays White
C_GRID, C_INK, C_MUTED = "#d9d9d9", "#333333", "#777777"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fig_colour(res, path):
    """
    Dot-and-whisker plot: Player 1 win rate with 95% Wilson CI, P1 as Black
    (circle) vs P1 as White (square), one pair per phase. Phases whose
    Holm-adjusted Fisher test is significant are marked with an asterisk.
    """
    n = len(res)
    x = np.arange(n)
    off = 0.17
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0.5, color=C_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)

    for sign, pre, colour, marker, label in (
            (-1, "black", C_BLACK, "o", "P1 plays Black (moves first)"),
            (+1, "white", C_WHITE, "s", "P1 plays White")):
        rate = res[f"rate_{pre}"].to_numpy()
        lo   = res[f"ci_{pre}_lo"].to_numpy()
        hi   = res[f"ci_{pre}_hi"].to_numpy()
        ax.errorbar(x + sign * off, rate, yerr=[np.clip(rate - lo, 0, None), np.clip(hi - rate, 0, None)],
                    fmt=marker, color=colour, markersize=6, elinewidth=1.6,
                    capsize=3, capthick=1.2, label=label, zorder=3)

    for i, sig in enumerate(res["significant_holm"]):
        if sig:
            ax.annotate("*", xy=(x[i], 1.025), ha="center", va="bottom",
                        fontsize=15, color=C_INK)

    labels = [short_label(p)[0] for p in res["phase"]]
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ph. {l}" for l in labels])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Player 1 win rate")
    ax.set_xlabel("Benchmark phase (* = significant colour effect, Holm-adjusted Fisher p < 0.05)")
    ax.legend(loc="lower left", ncol=1)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 6. Full results table and win-rate figure
# ---------------------------------------------------------------------------

def phase_name(phase):
    """'1. Random vs Random (sanity check)' -> 'Random vs Random'."""
    return re.sub(r"\s*\(.*?\)\s*$", "", short_label(phase)[1])


def full_results(df):
    out = []
    for phase, sub in df.groupby("phase", sort=False):
        W, D, L = counts(sub)
        n = W + D + L
        p, lo, hi = wilson_ci(W, n)
        out.append({
            "phase": phase, "num": short_label(phase)[0], "name": phase_name(phase),
            "n": n, "p1_win_pct": 100 * W / n, "draw_pct": 100 * D / n,
            "p2_win_pct": 100 * L / n,
            "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
            "verdict": verdict(lo, hi),
            "s_per_game": sub["duration_s"].mean(),
            "total_min": sub["duration_s"].sum() / 60,
            "cnn": "CNN" in phase,
        })
    return pd.DataFrame(out)


def latex_rows(fr, path):
    """Rows ready to paste into tab:full_results (same column order as the thesis)."""
    lines = []
    for i, r in fr.iterrows():
        if r["cnn"] and (i == 0 or not fr.loc[i - 1, "cnn"]):
            lines.append("\\addlinespace")
        name = r["name"].replace("Minimax AB CNN", "MinimaxAB-CNN").replace(
            "Minimax AB", "MinimaxAB").replace("Heuristic CNN", "HeuristicCNN").replace(
            "MCTS CNN", "MCTS-CNN").replace("Descent CNN", "Descent-CNN")
        lines.append(f"{r['num']}. {name} & {r['n']} & {r['p1_win_pct']:.1f} & "
                     f"{r['draw_pct']:.1f} & {r['p2_win_pct']:.1f} & "
                     f"[{r['ci_lo_pct']:.1f}, {r['ci_hi_pct']:.1f}] & "
                     f"{r['s_per_game']:.3f} \\\\")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


C_HEUR = "#2a78d6"      # blue   - phases with the handcrafted evaluator
C_CNN  = "#eb6834"      # orange - phases with the CNN evaluator


def fig_winrates(fr, path):
    """
    Player 1 win rate with 95% Wilson CI for every phase, one row per phase
    (top = Phase 1). Phases with the handcrafted evaluator (circles) and with
    the CNN evaluator (squares) are distinguished by colour and marker.
    """
    n = len(fr)
    y = np.arange(n)[::-1]                      # Phase 1 at the top
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.0))
    ax.axvline(50, color=C_MUTED, linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=0)

    for is_cnn, colour, marker, label in (
            (False, C_HEUR, "o", "Handcrafted evaluator (Phases 1–6)"),
            (True,  C_CNN,  "s", "CNN evaluator (Phases 8–11)")):
        m = fr["cnn"].to_numpy() == is_cnn
        rate = fr["p1_win_pct"].to_numpy()[m]
        lo, hi = fr["ci_lo_pct"].to_numpy()[m], fr["ci_hi_pct"].to_numpy()[m]
        ax.errorbar(rate, y[m], xerr=[np.clip(rate - lo, 0, None), np.clip(hi - rate, 0, None)],
                    fmt=marker, color=colour, markersize=6, elinewidth=1.6,
                    capsize=3, capthick=1.2, label=label, zorder=3)

    for yi, (_, r) in zip(y, fr.iterrows()):
        ax.annotate(f"{r['p1_win_pct']:.1f}%", xy=(r["ci_hi_pct"], yi),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=C_INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['num']}. {r['name']}" for _, r in fr.iterrows()])
    ax.tick_params(axis="y", length=0, labelcolor=C_INK)
    ax.set_xlim(0, 108)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Player 1 win rate (%) with 95% Wilson CI")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def _pct(x):
    return f"{100 * x:.1f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Othello benchmark post-hoc analysis")
    ap.add_argument("--report", nargs="+", default=[os.path.join(BENCH_DIR, "othello_benchmark_results.txt"),
                                                    os.path.join(BENCH_DIR, "othello_benchmark_results_cnn_v2.txt")],
                    help="one or more report files; if a phase appears in several "
                         "files, the one from the LAST file is used")
    ap.add_argument("--out",    default=os.path.join(BENCH_DIR, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = []
    for path in args.report:
        part = parse_report(path)
        # a phase re-run in a later file replaces the same phase from earlier files
        frames = [f[~f["phase"].isin(part["phase"].unique())] for f in frames]
        frames.append(part)
        print(f"  {path}: {part['phase'].nunique()} phases")
    df = pd.concat(frames, ignore_index=True)
    order = {p: int(short_label(p)[0]) for p in df["phase"].unique()}
    df = df.sort_values("phase", key=lambda s: s.map(order), kind="stable")
    print(f"Parsed {len(df)} games in {df['phase'].nunique()} phases\n")

    # --- Draw sensitivity ---
    ds = draw_sensitivity(df)
    ds.to_csv(os.path.join(args.out, "draws_sensitivity.csv"), index=False)
    print("DRAW SENSITIVITY  (a) W/n  vs  (b) W/(W+L)")
    for _, r in ds.iterrows():
        print(f"  {r['phase']:<36} D={r['D']:>3}  "
              f"(a) {_pct(r['rate_all']):>5} [{_pct(r['ci_all_lo'])}, {_pct(r['ci_all_hi'])}] {r['verdict_all']:<9}  "
              f"(b) {_pct(r['rate_dec']):>5} [{_pct(r['ci_dec_lo'])}, {_pct(r['ci_dec_hi'])}] {r['verdict_dec']}")
    print(f"  Verdicts changed: {int(ds['verdict_changed'].sum())} of {len(ds)}\n")

    # --- Colour effects ---
    ce, pooled = colour_effects(df)
    ce.to_csv(os.path.join(args.out, "colour_effects.csv"), index=False)
    print("COLOUR EFFECTS - Model 1: per-phase Fisher exact test")
    for _, r in ce.iterrows():
        print(f"  {r['phase']:<36} Black {r['W_black']:>3}/{r['n_black']:<3} "
              f"White {r['W_white']:>3}/{r['n_white']:<3}  OR={r['odds_ratio']:<8.3g} "
              f"p={r['fisher_p']:.2e}  p_Holm={r['fisher_p_holm']:.2e}  "
              f"Black-side {_pct(r['black_side_rate'])}%")
    print("\nCOLOUR EFFECTS - Model 2: Cochran-Mantel-Haenszel across phases")
    print(f"  Pooled OR = {pooled['pooled_OR']:.3f}  "
          f"95% CI [{pooled['pooled_OR_lo']:.3f}, {pooled['pooled_OR_hi']:.3f}]")
    print(f"  CMH test (common OR = 1): chi2 = {pooled['cmh_stat']:.3f}, p = {pooled['cmh_p']:.3g}")
    print(f"  Breslow-Day (ORs homogeneous): chi2 = {pooled['bd_stat']:.3f}, p = {pooled['bd_p']:.3g}")
    if "excl_n_phases" in pooled:
        print(f"  Without Holm-significant phases ({pooled['excl_n_phases']} phases left): "
              f"pooled OR = {pooled['excl_pooled_OR']:.3f}, CMH p = {pooled['excl_cmh_p']:.3g}, "
              f"Breslow-Day p = {pooled['excl_bd_p']:.3g}")
    pd.DataFrame([pooled]).to_csv(os.path.join(args.out, "colour_pooled_tests.csv"), index=False)

    # --- Full results table + win-rate figure ---
    fr = full_results(df)
    fr.to_csv(os.path.join(args.out, "full_results.csv"), index=False)
    print("\nFULL RESULTS (draws = non-wins)")
    for _, r in fr.iterrows():
        print(f"  {r['phase']:<36} N={r['n']:>3}  P1 {r['p1_win_pct']:5.1f}  D {r['draw_pct']:4.1f}  "
              f"P2 {r['p2_win_pct']:5.1f}  CI [{r['ci_lo_pct']:.1f}, {r['ci_hi_pct']:.1f}]  "
              f"{r['s_per_game']:.3f} s/game  ({r['total_min']:.1f} min)  {r['verdict']}")
    print(f"  Total measured game time: {fr['total_min'].sum():.1f} min")

    print()
    latex_rows(fr, os.path.join(args.out, "full_results_rows.tex"))
    fig_winrates(fr, os.path.join(args.out, "winrates.png"))
    fig_colour(ce, os.path.join(args.out, "fig_colour_effects.png"))