import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
RED = "#e34948"
GOOD_TEXT = "#006300"
CRITICAL = "#d03b3b"
FONT = ["system-ui", "Segoe UI", "DejaVu Sans"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT

GROUPS = {
    "Contract": ["Contract_One Year", "Contract_Two Year"],
    "Tenure": ["Tenure in Months"],
    "Monthly charge": ["Monthly Charge"],
    "Internet type": "Internet Type_",
    "Payment method": "Payment Method_",
    "Age": ["Age"],
    "Dependents": ["Number of Dependents"],
    "Referrals": ["Number of Referrals", "Referred a Friend"],
    "Download volume": ["Avg Monthly GB Download"],
    "Paperless billing": ["Paperless Billing"],
}


def group_contributions(columns, values):
    values = np.asarray(values, dtype=float).ravel()
    index = {c: i for i, c in enumerate(columns)}
    used = set()
    result = {}
    for name, spec in GROUPS.items():
        if isinstance(spec, str):
            cols = [c for c in columns if c.startswith(spec)]
        else:
            cols = [c for c in spec if c in index]
        used.update(cols)
        result[name] = float(sum(values[index[c]] for c in cols))
    rest = [c for c in columns if c not in used]
    result["Other variables"] = float(sum(values[index[c]] for c in rest))
    return result


def _panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def _card(ax):
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=1, edgecolor=GRID, facecolor=SURFACE, transform=ax.transAxes, clip_on=False,
    ))


def build_recap(profile, proba, threshold, expected_gain, contact, assumptions, contributions):
    fig = plt.figure(figsize=(11, 7.6), facecolor=SURFACE)

    fig.text(0.05, 0.945, "Customer churn recap", fontsize=20, fontweight="bold", color=INK, va="center")
    fig.text(
        0.05, 0.905,
        f"Demo on the IBM Telco dataset (fictional)  ·  {date.today().strftime('%d %b %Y')}",
        fontsize=9.5, color=MUTED, va="center",
    )

    hero = _panel(fig, [0.05, 0.64, 0.40, 0.22])
    _card(hero)
    hero.text(0.06, 0.88, "Probability of leaving", fontsize=10.5, color=INK_2, va="center", transform=hero.transAxes)
    hero.text(0.06, 0.60, f"{proba:.0%}", fontsize=44, fontweight="bold", color=INK, va="center", transform=hero.transAxes)
    track_y, track_h = 0.26, 0.11
    hero.add_patch(Rectangle((0.06, track_y), 0.88, track_h, color=GRID, transform=hero.transAxes))
    hero.add_patch(Rectangle((0.06, track_y), 0.88 * min(proba, 1.0), track_h, color=BLUE, transform=hero.transAxes))
    if threshold <= 1:
        tx = 0.06 + 0.88 * threshold
        hero.plot([tx, tx], [track_y - 0.05, track_y + track_h + 0.05], color=INK, linewidth=1.6, transform=hero.transAxes)
        hero.text(tx, track_y - 0.09, f"Break-even {threshold:.0%}", fontsize=8.5, color=INK_2,
                  ha="center", va="top", transform=hero.transAxes)

    decision = _panel(fig, [0.50, 0.64, 0.45, 0.22])
    _card(decision)
    if contact:
        decision.text(0.06, 0.82, "✔  Contact with a retention offer", fontsize=13, fontweight="bold",
                      color=GOOD_TEXT, va="center", transform=decision.transAxes)
    else:
        decision.text(0.06, 0.82, "✖  No action", fontsize=13, fontweight="bold",
                      color=INK_2, va="center", transform=decision.transAxes)
    gain_label = "Expected gain" if expected_gain >= 0 else "Expected cost of contacting"
    decision.text(0.06, 0.55, f"{gain_label}", fontsize=10, color=INK_2, va="center", transform=decision.transAxes)
    decision.text(0.06, 0.34, f"{abs(expected_gain):,.0f} €", fontsize=26, fontweight="bold", color=INK,
                  va="center", transform=decision.transAxes)
    decision.text(
        0.06, 0.10,
        f"offer {assumptions['offer_cost']:,.0f} €  ·  success rate {assumptions['success_rate']:.0%}  ·  "
        f"customer value {assumptions['customer_value']:,.0f} €",
        fontsize=8.5, color=MUTED, va="center", transform=decision.transAxes,
    )

    prof = _panel(fig, [0.05, 0.14, 0.27, 0.44])
    prof.text(0.0, 0.97, "Customer profile", fontsize=11.5, fontweight="bold", color=INK, va="center")
    n = len(profile)
    for i, (label, value) in enumerate(profile):
        y = 0.86 - i * (0.80 / max(n - 1, 1))
        prof.text(0.0, y, label, fontsize=9.5, color=INK_2, va="center")
        prof.text(1.0, y, value, fontsize=9.5, color=INK, va="center", ha="right")
        prof.plot([0, 1], [y - 0.04, y - 0.04], color=GRID, linewidth=0.8)

    ax = fig.add_axes([0.46, 0.14, 0.40, 0.38], facecolor=SURFACE)
    fig.text(0.37, 0.567, "What moves this score", fontsize=11.5, fontweight="bold", color=INK, va="center")
    fig.text(0.37, 0.537, "Push of each answer compared with the average customer", fontsize=8.5, color=MUTED, va="center")

    items = [(k, v) for k, v in contributions.items() if k != "Other variables"]
    items.sort(key=lambda kv: abs(kv[1]))
    if "Other variables" in contributions:
        items.insert(0, ("Other variables", contributions["Other variables"]))
    names = [k for k, _ in items]
    vals = np.array([v for _, v in items])
    ypos = np.arange(len(items))
    colors = [RED if v > 0 else BLUE for v in vals]
    ax.barh(ypos, vals, height=0.55, color=colors, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(0, color=AXIS, linewidth=1)
    span = max(np.abs(vals).max(), 0.5)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5, color=INK_2)
    for y, v in zip(ypos, vals):
        offset = span * 0.04
        text = "0.00" if round(v, 2) == 0 else f"{v:+.2f}".replace("-", "−")
        ax.text(v + (offset if v >= 0 else -offset), y, text, fontsize=8.5, color=INK,
                va="center", ha="left" if v >= 0 else "right")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    lx = 0.37
    for color, label, dx in ((RED, "Raises the risk", 0.0), (BLUE, "Lowers the risk", 0.15)):
        fig.patches.append(Rectangle((lx + dx, 0.083), 0.012, 0.018, transform=fig.transFigure, color=color))
        fig.text(lx + dx + 0.018, 0.092, label, fontsize=8.5, color=INK_2, va="center")

    fig.text(
        0.05, 0.035,
        "Scores come from a gradient boosting model trained on about 6,000 fictional customers with more than 3 months of tenure. "
        "Pushes are SHAP values (log-odds).\nOnly 10 answers are asked; the other variables take typical values. "
        "The costs are assumptions. Not for real decisions.",
        fontsize=7.5, color=MUTED, va="center", linespacing=1.5,
    )
    return fig


def figure_bytes(fig, fmt):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=160, facecolor=fig.get_facecolor())
    return buffer.getvalue()
