"""
Generate classic CNN-architecture figures (3-D coloured blocks) for each Stage.
Improved version: larger blocks, readable labels, no overlap.

Run:
    python3 /tmp/gen_arch_figs/gen_arch_figs.py

Outputs (300 dpi PNG):
    /tmp/gen_arch_figs/stage1_architecture.png
    /tmp/gen_arch_figs/stage2_architecture.png
    /tmp/gen_arch_figs/stage3_architecture.png
"""

from pathlib import Path
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C = dict(
    input  ="#BDBDBD",
    conv   ="#4472C4",
    relu   ="#E06C6C",
    pool   ="#F0C040",
    se     ="#ED7D31",
    spattn ="#C0504D",
    adapter="#70AD47",
    fc     ="#5B9BD5",
    output ="#9B59B6",
    frozen ="#A8A8A8",
)

# ── Colour helpers ─────────────────────────────────────────────────────────

def _h2r(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))

def _lighten(hex_c, a=0.28):
    h, s, v = colorsys.rgb_to_hsv(*_h2r(hex_c))
    v = min(1.0, v+a)
    return "#{:02x}{:02x}{:02x}".format(*[int(x*255) for x in colorsys.hsv_to_rgb(h,s,v)])

def _darken(hex_c, a=0.20):
    h, s, v = colorsys.rgb_to_hsv(*_h2r(hex_c))
    v = max(0.0, v-a)
    return "#{:02x}{:02x}{:02x}".format(*[int(x*255) for x in colorsys.hsv_to_rgb(h,s,v)])

# ── 3-D block ──────────────────────────────────────────────────────────────

def block(ax, x, y_center, w, h, d, color, alpha=1.0, lw=0.9):
    """Draw one 3-D slab; bottom-left front corner = (x, y_center - h/2)."""
    y = y_center - h/2
    ox, oy = d*0.50, d*0.38
    front = plt.Polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],
                        closed=True, fc=color, ec="#222", lw=lw, alpha=alpha, zorder=3)
    top   = plt.Polygon([[x,y+h],[x+w,y+h],[x+w+ox,y+h+oy],[x+ox,y+h+oy]],
                        closed=True, fc=_lighten(color), ec="#222", lw=lw, alpha=alpha, zorder=3)
    right = plt.Polygon([[x+w,y],[x+w+ox,y+oy],[x+w+ox,y+h+oy],[x+w,y+h]],
                        closed=True, fc=_darken(color), ec="#222", lw=lw, alpha=alpha, zorder=3)
    for p in (front, top, right):
        ax.add_patch(p)
    # exit = mid-right
    return x+w + ox*0.5, y_center + oy*0.25

def arrow(ax, x0, y0, x1, y1, col="#333"):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=12),
                zorder=10)

def txt(ax, x, y, s, fs=7.5, ha="center", va="center", bold=False, col="#111"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=col, zorder=12,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def badge(ax, x, y, label, fc="#777", col="white", fs=6.5):
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=col, zorder=14,
            bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec="none", alpha=0.88))

def legend(ax, entries, y_anchor=-0.03, ncol=None):
    patches = [mpatches.Patch(fc=c, ec="#444", label=l) for c,l in entries]
    ax.legend(handles=patches, loc="lower left",
              bbox_to_anchor=(0.0, y_anchor),
              ncol=ncol or len(entries),
              frameon=True, fontsize=7,
              handlelength=1.3, handleheight=1.0,
              borderpad=0.5, columnspacing=0.9)

# ── One-pass sequential layer renderer ─────────────────────────────────────

def render_layers(ax, layers, x0, y0, gap):
    """
    layers: list of dicts with keys w,h,d,color,top_label,bot_label,top_offset
    Returns list of (x, exit_x, exit_y) per layer, and final x.
    """
    x = x0
    prev_ex, prev_ey = x0 - gap, y0
    results = []
    for L in layers:
        d  = L.get("d", 0.22)
        ex, ey = block(ax, x, y0, L["w"], L["h"], d, L["color"])
        cx = x + L["w"]/2

        top = L.get("top_label", "")
        bot = L.get("bot_label", "")
        top_off = L.get("top_offset", 0.08)
        bot_off = L.get("bot_offset", 0.08)

        if top:
            txt(ax, cx, y0 + L["h"]/2 + d*0.38 + top_off, top, fs=7.2)
        if bot:
            txt(ax, cx, y0 - L["h"]/2 - bot_off, bot, fs=6.5, col="#555")

        arrow(ax, prev_ex, prev_ey, x, y0)
        prev_ex, prev_ey = ex, ey
        results.append(dict(x=x, w=L["w"], h=L["h"], d=d, ex=ex, ey=ey))
        x += L["w"] + gap
    return results, x - gap


# ── Backbone spec builder ───────────────────────────────────────────────────

def backbone_layers(frozen=False):
    cb = C["frozen"] if frozen else C["conv"]
    D  = 0.24   # depth of backbone blocks

    return [
        dict(w=0.55, h=2.20, d=D, color=cb,       top_label="Conv\n7×7/2", bot_label="64ch"),
        dict(w=0.30, h=2.20, d=D, color=C["relu"], top_label="BN\nReLU"),
        dict(w=0.36, h=1.80, d=D, color=C["pool"], top_label="MaxPool\n/2"),
        dict(w=0.60, h=1.80, d=D, color=cb,        top_label="Layer1\n×2",  bot_label="64ch"),
        dict(w=0.30, h=0.65, d=D, color=C["se"],   top_label="SE\n(64)"),
        dict(w=0.60, h=1.50, d=D, color=cb,        top_label="Layer2\n×2",  bot_label="128ch"),
        dict(w=0.30, h=0.65, d=D, color=C["se"],   top_label="SE\n(128)"),
        dict(w=0.60, h=1.22, d=D, color=cb,        top_label="Layer3\n×2",  bot_label="256ch"),
        dict(w=0.30, h=0.65, d=D, color=C["se"],   top_label="SE\n(256)"),
        dict(w=0.60, h=0.98, d=D, color=cb,        top_label="Layer4\n×2",  bot_label="512ch"),
        dict(w=0.30, h=0.65, d=D, color=C["se"],   top_label="SE\n(512)"),
        dict(w=0.30, h=0.65, d=D, color=C["spattn"],top_label="Spatial\nAttn"),
        dict(w=0.30, h=0.55, d=D, color=C["pool"], top_label="AvgPool"),
        dict(w=0.30, h=0.55, d=D, color=C["pool"], top_label="Flatten\n512-d"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════

def stage1():
    fig, ax = plt.subplots(figsize=(22, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")
    mid = 0.0

    # ── Input ──────────────────────────────────────────────────────────────
    block(ax, 0.0, mid, 0.38, 2.20, 0.24, C["input"])
    txt(ax, 0.19, mid - 1.28, "Input\n1×H×W", fs=7.5)

    # ── Backbone (fully trainable) ─────────────────────────────────────────
    bb_layers = backbone_layers(frozen=False)
    GAP = 0.36
    results, x_end = render_layers(ax, bb_layers, x0=0.80, y0=mid, gap=GAP)
    arrow(ax, 0.38 + 0.24*0.5, mid, 0.80, mid)   # input → first layer

    # ── Head 1 ─────────────────────────────────────────────────────────────
    x_h = x_end + 0.55
    arrow(ax, results[-1]["ex"], results[-1]["ey"], x_h, mid)

    head = [
        dict(w=0.45, h=0.90, color=C["fc"],     top_label="FC\n512→256"),
        dict(w=0.28, h=0.65, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.26, h=0.65, color=C["relu"],   top_label="Drop\n0.3"),
        dict(w=0.40, h=0.65, color=C["fc"],     top_label="FC\n256→2"),
        dict(w=0.34, h=0.65, color=C["output"], top_label="Softmax"),
    ]
    h_results, hx_end = render_layers(ax, head, x0=x_h, y0=mid, gap=0.26)
    arrow(ax, h_results[-1]["ex"], h_results[-1]["ey"], hx_end + 0.28, mid)
    txt(ax, hx_end + 0.55, mid, "INTRA\nvs\nNON-INTRA",
        fs=9, col="#6C0BA9", bold=True)

    # ── Axis limits ────────────────────────────────────────────────────────
    ax.set_xlim(-0.3, hx_end + 1.6)
    ax.set_ylim(-1.8, 2.4)

    ax.set_title("Stage 1 — Full Backbone Training  (Focal Loss γ=2, α=0.25)",
                 fontsize=13, fontweight="bold", pad=10)

    legend(ax, [
        (C["input"],  "Input"),
        (C["conv"],   "Conv / ResLayer"),
        (C["relu"],   "BN / ReLU / Dropout"),
        (C["pool"],   "Pool / Flatten"),
        (C["se"],     "SE-Block"),
        (C["spattn"], "SpatialAttention"),
        (C["fc"],     "FC Layer"),
        (C["output"], "Prediction"),
    ])

    fig.tight_layout()
    fig.savefig(OUT / "stage1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓  stage1_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2
# ══════════════════════════════════════════════════════════════════════════════

def stage2():
    fig, ax = plt.subplots(figsize=(26, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    mid = 0.0

    # ── Input ──────────────────────────────────────────────────────────────
    block(ax, 0.0, mid, 0.38, 2.20, 0.24, C["input"])
    txt(ax, 0.19, mid - 1.28, "Input\n1×H×W", fs=7.5)

    # ── Frozen backbone ────────────────────────────────────────────────────
    bb_layers = backbone_layers(frozen=True)
    GAP = 0.36
    results, x_end = render_layers(ax, bb_layers, x0=0.80, y0=mid, gap=GAP)
    arrow(ax, 0.38 + 0.24*0.5, mid, 0.80, mid)

    # ── FROZEN badge ───────────────────────────────────────────────────────
    bb_cx = (0.80 + x_end) / 2
    badge(ax, bb_cx, mid + 1.55, "❄  FROZEN BACKBONE")

    # ── Conv-Adapter3 (after Layer3 = index 8, SE(256)) ────────────────────
    # Layer3 block is at index 7, SE(256) at index 8
    se3 = results[8]   # SE(256)
    a3x = se3["x"] + se3["w"] + GAP*0.05
    a3y_ctr = mid - 1.65
    block(ax, a3x, a3y_ctr, 0.42, 0.82, 0.18, C["adapter"])
    txt(ax, a3x + 0.21, a3y_ctr - 0.56, "ConvAdapter3\n(256ch)", fs=6.8)
    # down: from bottom of SE(256), up: to Layer4 input
    arrow(ax, se3["x"] + se3["w"]/2, mid - se3["h"]/2,
          a3x + 0.21, a3y_ctr + 0.41, col="#555")
    layer4_x = results[9]["x"]
    arrow(ax, a3x + 0.42 + 0.18*0.5, a3y_ctr,
          layer4_x, mid, col="#555")

    # ── Conv-Adapter4 (after Layer4 = index 9) ─────────────────────────────
    l4 = results[9]   # Layer4
    a4x = l4["x"] + l4["w"] + GAP*0.05
    a4y_ctr = mid + 1.45
    block(ax, a4x, a4y_ctr, 0.42, 0.82, 0.18, C["adapter"])
    txt(ax, a4x + 0.21, a4y_ctr + 0.68, "ConvAdapter4\n(512ch)", fs=6.8)
    arrow(ax, l4["x"] + l4["w"]/2, mid + l4["h"]/2,
          a4x + 0.21, a4y_ctr - 0.41, col="#555")
    se4_x = results[10]["x"]
    arrow(ax, a4x + 0.42 + 0.18*0.5, a4y_ctr,
          se4_x, mid, col="#555")

    # ── Head 2 ─────────────────────────────────────────────────────────────
    x_h = x_end + 0.55
    arrow(ax, results[-1]["ex"], results[-1]["ey"], x_h, mid)

    head = [
        dict(w=0.45, h=0.90, color=C["fc"],     top_label="FC\n512→256"),
        dict(w=0.28, h=0.65, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.26, h=0.65, color=C["relu"],   top_label="Drop\n0.4"),
        dict(w=0.45, h=0.65, color=C["fc"],     top_label="FC\n256→128"),
        dict(w=0.28, h=0.65, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.26, h=0.65, color=C["relu"],   top_label="Drop\n0.4"),
        dict(w=0.40, h=0.65, color=C["fc"],     top_label="FC\n128→3"),
        dict(w=0.34, h=0.65, color=C["output"], top_label="Softmax"),
    ]
    h_results, hx_end = render_layers(ax, head, x0=x_h, y0=mid, gap=0.24)
    arrow(ax, h_results[-1]["ex"], h_results[-1]["ey"], hx_end + 0.28, mid)
    txt(ax, hx_end + 0.60, mid, "DC-ONLY\nvs\nAB-ONLY\nvs\nBOTH",
        fs=9, col="#6C0BA9", bold=True)

    ax.set_xlim(-0.3, hx_end + 2.0)
    ax.set_ylim(-2.8, 2.8)

    ax.set_title(
        "Stage 2 — Conv-Adapters with Frozen Backbone  (CB-Focal Loss β=0.9999, γ=2)",
        fontsize=13, fontweight="bold", pad=10)

    legend(ax, [
        (C["input"],   "Input"),
        (C["frozen"],  "Frozen Backbone"),
        (C["relu"],    "BN / ReLU / Dropout"),
        (C["pool"],    "Pool / Flatten"),
        (C["se"],      "SE-Block"),
        (C["adapter"], "Conv-Adapter (trainable)"),
        (C["fc"],      "FC Layer"),
        (C["output"],  "Prediction"),
    ])

    fig.tight_layout()
    fig.savefig(OUT / "stage2_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓  stage2_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3
# ══════════════════════════════════════════════════════════════════════════════

def stage3():
    fig, ax = plt.subplots(figsize=(26, 8.5))
    ax.set_aspect("equal")
    ax.axis("off")
    mid = 0.0

    # ── Input ──────────────────────────────────────────────────────────────
    block(ax, 0.0, mid, 0.38, 2.20, 0.24, C["input"])
    txt(ax, 0.19, mid - 1.28, "Input\n1×H×W", fs=7.5)

    # ── Frozen AdapterBackbone ─────────────────────────────────────────────
    bb_layers = backbone_layers(frozen=True)
    GAP = 0.36
    results, x_end = render_layers(ax, bb_layers, x0=0.80, y0=mid, gap=GAP)
    arrow(ax, 0.38 + 0.24*0.5, mid, 0.80, mid)

    bb_cx = (0.80 + x_end) / 2
    badge(ax, bb_cx, mid + 1.55, "❄  FROZEN  (backbone + adapters)")

    # ── Small adapter icons (just icons, no arrows, to show they exist) ────
    se3 = results[8]
    a3_icon_x = se3["x"] + se3["w"] + GAP*0.05
    block(ax, a3_icon_x, mid - 1.35, 0.30, 0.55, 0.14, C["adapter"])
    txt(ax, a3_icon_x + 0.15, mid - 1.72, "Adapter3", fs=6.2, col="#444")
    arrow(ax, se3["x"] + se3["w"]/2, mid - se3["h"]/2,
          a3_icon_x + 0.15, mid - 1.35 + 0.275, col="#888")
    arrow(ax, a3_icon_x + 0.30 + 0.14*0.5, mid - 1.35 + 0.275,
          results[9]["x"], mid, col="#888")

    l4 = results[9]
    a4_icon_x = l4["x"] + l4["w"] + GAP*0.05
    block(ax, a4_icon_x, mid + 1.15, 0.30, 0.55, 0.14, C["adapter"])
    txt(ax, a4_icon_x + 0.15, mid + 1.55, "Adapter4", fs=6.2, col="#444")
    arrow(ax, l4["x"] + l4["w"]/2, mid + l4["h"]/2,
          a4_icon_x + 0.15, mid + 1.15, col="#888")
    arrow(ax, a4_icon_x + 0.30 + 0.14*0.5, mid + 1.15 + 0.275,
          results[10]["x"], mid, col="#888")

    # ── Fork ───────────────────────────────────────────────────────────────
    fork_x = x_end + 0.50
    # horizontal line from last block to fork
    ax.plot([results[-1]["ex"], fork_x], [mid, mid], "k-", lw=1.2, zorder=5)
    # vertical line
    y_rect = 2.50
    y_ab   = -2.50
    ax.plot([fork_x, fork_x], [y_ab, y_rect], "k-", lw=1.2, zorder=5)

    # ── Head 3-RECT (top branch) ───────────────────────────────────────────
    arrow(ax, fork_x, y_rect, fork_x + 0.28, y_rect)
    rect_head = [
        dict(w=0.44, h=0.85, color=C["fc"],     top_label="FC\n512→128"),
        dict(w=0.26, h=0.62, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.24, h=0.62, color=C["relu"],   top_label="Drop\n0.2"),
        dict(w=0.44, h=0.62, color=C["fc"],     top_label="FC\n128→64"),
        dict(w=0.26, h=0.62, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.38, h=0.62, color=C["fc"],     top_label="FC\n64→2"),
        dict(w=0.32, h=0.62, color=C["output"], top_label="Softmax"),
    ]
    rh, rx_end = render_layers(ax, rect_head, x0=fork_x+0.28, y0=y_rect, gap=0.24)
    arrow(ax, rh[-1]["ex"], rh[-1]["ey"], rx_end + 0.28, y_rect)
    txt(ax, rx_end + 0.62, y_rect, "RECT\nvs NON-RECT",
        fs=9, col="#6C0BA9", bold=True)
    txt(ax, fork_x + 0.30, y_rect + 0.72,
        "Head 3-RECT  (Focal Loss γ=2.0)",
        fs=9, bold=True, col="#2E4057", ha="left")

    # ── Head 3-AB (bottom branch) ──────────────────────────────────────────
    arrow(ax, fork_x, y_ab, fork_x + 0.28, y_ab)
    ab_head = [
        dict(w=0.44, h=0.85, color=C["fc"],     top_label="FC\n512→256"),
        dict(w=0.26, h=0.62, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.24, h=0.62, color=C["relu"],   top_label="Drop\n0.5"),
        dict(w=0.44, h=0.62, color=C["fc"],     top_label="FC\n256→128"),
        dict(w=0.26, h=0.62, color=C["relu"],   top_label="BN\nReLU"),
        dict(w=0.38, h=0.62, color=C["fc"],     top_label="FC\n128→4"),
        dict(w=0.32, h=0.62, color=C["output"], top_label="Softmax"),
    ]
    ah, ax_end = render_layers(ax, ab_head, x0=fork_x+0.28, y0=y_ab, gap=0.24)
    arrow(ax, ah[-1]["ex"], ah[-1]["ey"], ax_end + 0.28, y_ab)
    txt(ax, ax_end + 0.58, y_ab, "AB\nvs NON-AB",
        fs=9, col="#6C0BA9", bold=True)
    txt(ax, fork_x + 0.30, y_ab - 0.70,
        "Head 3-AB  (Focal Loss γ=2.5)",
        fs=9, bold=True, col="#2E4057", ha="left")

    xlim_r = max(rx_end, ax_end) + 1.6
    ax.set_xlim(-0.3, xlim_r)
    ax.set_ylim(y_ab - 1.4, y_rect + 1.4)

    ax.set_title(
        "Stage 3 — Specialised Heads  (backbone + adapters frozen)",
        fontsize=13, fontweight="bold", pad=10)

    legend(ax, [
        (C["input"],   "Input"),
        (C["frozen"],  "Frozen Backbone"),
        (C["relu"],    "BN / ReLU / Dropout"),
        (C["pool"],    "Pool / Flatten"),
        (C["se"],      "SE-Block"),
        (C["adapter"], "Conv-Adapter (frozen)"),
        (C["fc"],      "FC Layer"),
        (C["output"],  "Prediction"),
    ], y_anchor=-0.04)

    fig.tight_layout()
    fig.savefig(OUT / "stage3_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓  stage3_architecture.png")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stage1()
    stage2()
    stage3()
    print("\nAll figures saved to", OUT)
