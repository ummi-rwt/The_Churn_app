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
BLUE = "#2a78d6"
RED = "#e34948"
AMBER = "#c77d0a"
GREEN = "#1b7a3d"
FONT = ["system-ui", "Segoe UI", "DejaVu Sans"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = FONT

GROUPS = {
    "Contract type": ["Contract_One Year", "Contract_Two Year"],
    "Time as a customer": ["Tenure in Months"],
    "Monthly bill": ["Monthly Charge"],
    "Internet connection": "Internet Type_",
    "Payment method": "Payment Method_",
    "Age": ["Age"],
    "People in the household": ["Number of Dependents"],
    "Friends referred": ["Number of Referrals", "Referred a Friend"],
    "Data usage": ["Avg Monthly GB Download"],
    "Online-only billing": ["Paperless Billing"],
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
    result["Other factors"] = float(sum(values[index[c]] for c in rest))
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


def _level(proba, threshold):
    if threshold <= 1:
        if proba >= threshold:
            return "HIGH RISK", RED
        if proba >= 0.6 * threshold:
            return "MODERATE RISK", AMBER
        return "LOW RISK", GREEN
    if proba >= 0.5:
        return "HIGH RISK", RED
    if proba >= 0.25:
        return "MODERATE RISK", AMBER
    return "LOW RISK", GREEN


def _short(value):
    return f" ({value})" if value else ""


def build_recap(profile, proba, threshold, expected_gain, contact, assumptions, contributions):
    fig = plt.figure(figsize=(11, 7.4), facecolor=SURFACE)

    fig.text(0.05, 0.94, "Customer summary", fontsize=20, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.90, f"Demo with fictional data  ·  {date.today().strftime('%d %b %Y')}",
             fontsize=9.5, color=MUTED, va="center")

    risk = _panel(fig, [0.05, 0.60, 0.43, 0.25])
    _card(risk)
    level, level_color = _level(proba, threshold)
    risk.text(0.06, 0.86, "Chance this customer leaves", fontsize=10.5, color=INK_2, va="center")
    risk.text(0.06, 0.62, f"{proba:.0%}", fontsize=40, fontweight="bold", color=INK, va="center")
    risk.text(0.94, 0.62, level, fontsize=13, fontweight="bold", color=level_color, va="center", ha="right")
    track_y, track_h = 0.27, 0.11
    risk.add_patch(Rectangle((0.06, track_y), 0.88, track_h, color=GRID))
    risk.add_patch(Rectangle((0.06, track_y), 0.88 * min(proba, 1.0), track_h, color=level_color))
    if threshold <= 1:
        tx = 0.06 + 0.88 * threshold
        risk.plot([tx, tx], [track_y - 0.05, track_y + track_h + 0.05], color=INK, linewidth=1.6)
        risk.text(tx, track_y - 0.09, f"Worth acting above {threshold:.0%}", fontsize=8.5, color=INK_2,
                  ha="center", va="top")

    action = _panel(fig, [0.52, 0.60, 0.43, 0.25])
    _card(action)
    action.text(0.06, 0.86, "What to do", fontsize=10.5, color=INK_2, va="center")
    if contact:
        action.text(0.06, 0.66, "Make an offer to keep them", fontsize=16, fontweight="bold", color=GREEN, va="center")
        lines = ("Expected result: about", f"{abs(expected_gain):,.0f} € gained on average.")
    else:
        action.text(0.06, 0.66, "No action needed for now", fontsize=16, fontweight="bold", color=INK_2, va="center")
        lines = ("The offer would cost about", f"{abs(expected_gain):,.0f} € more than it brings back.")
    action.text(0.06, 0.48, lines[0], fontsize=9.5, color=INK, va="center")
    action.text(0.06, 0.38, lines[1], fontsize=9.5, color=INK, va="center")
    action.text(
        0.06, 0.20,
        f"We assumed: the offer costs {assumptions['offer_cost']:,.0f} € and works "
        f"{assumptions['success_rate']:.0%} of the time,",
        fontsize=8.5, color=MUTED, va="center",
    )
    action.text(
        0.06, 0.10,
        f"and a customer who stays is worth {assumptions['customer_value']:,.0f} € a year.",
        fontsize=8.5, color=MUTED, va="center",
    )

    snap = _panel(fig, [0.05, 0.08, 0.30, 0.44])
    snap.text(0.0, 0.97, "Customer details", fontsize=12, fontweight="bold", color=INK, va="center")
    n = len(profile)
    for i, (label, value) in enumerate(profile):
        y = 0.86 - i * (0.80 / max(n - 1, 1))
        snap.text(0.0, y, label, fontsize=9.5, color=INK_2, va="center")
        snap.text(1.0, y, value, fontsize=9.5, color=INK, va="center", ha="right")
        snap.plot([0, 1], [y - 0.04, y - 0.04], color=GRID, linewidth=0.8)

    names = list(GROUPS.keys())
    values = [value for _, value in profile] if len(profile) == len(GROUPS) else [None] * len(GROUPS)
    shown = dict(zip(names, values))
    items = [(k, contributions.get(k, 0.0)) for k in names]
    items.sort(key=lambda kv: kv[1])
    up = max(items, key=lambda kv: kv[1])
    down = min(items, key=lambda kv: kv[1])

    fig.text(0.40, 0.515, "What explains this result", fontsize=12, fontweight="bold", color=INK, va="center")
    fig.text(0.40, 0.482, "Each bar shows how much one detail raises or lowers the risk.", fontsize=9, color=MUTED, va="center")
    reasons = []
    if up[1] >= 0.1:
        reasons.append(("Main reason they might leave: ", f"{up[0]}{_short(shown.get(up[0]))}", RED))
    if down[1] <= -0.1:
        reasons.append(("Main reason they might stay: ", f"{down[0]}{_short(shown.get(down[0]))}", BLUE))
    for k, (lead, text, color) in enumerate(reasons):
        y = 0.445 - k * 0.03
        fig.text(0.40, y, lead, fontsize=9.5, color=INK_2, va="center")
        fig.text(0.60, y, text, fontsize=9.5, fontweight="bold", color=color, va="center")

    ax = fig.add_axes([0.66, 0.12, 0.29, 0.25], facecolor=SURFACE)
    vals = np.array([v for _, v in items])
    ypos = np.arange(len(items))
    ax.barh(ypos, vals, height=0.62, color=[RED if v > 0 else BLUE for v in vals], edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=MUTED, linewidth=1)
    span = max(np.abs(vals).max(), 0.5) * 1.1
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{k}: {shown[k]}" if shown.get(k) else k for k, _ in items], fontsize=9, color=INK_2)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    fig.text(0.66, 0.085, "← Helps keep them", fontsize=9, color=BLUE, fontweight="bold", va="center")
    fig.text(0.95, 0.085, "Pushes them to leave →", fontsize=9, color=RED, fontweight="bold", va="center", ha="right")

    return fig


def figure_bytes(fig, fmt):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=160, facecolor=fig.get_facecolor())
    return buffer.getvalue()
