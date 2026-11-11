#!/usr/bin/env python3
"""Algorithm flowchart of the attention holding (ISO 5807 / ДСТУ 19.701 symbols):
terminator, input/output (parallelogram), process (rectangle), decision (diamond).
English labels for the AAIT article 4b (abductive identity in the gate), Times New Roman 14 pt, 600 dpi -> fig/en/fig_algo_en.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import A4   # noqa: E402  (works in article4/code and in the repository attention/)
FIG = os.path.join(A4, "fig")
TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
font_manager.fontManager.addfont(TNR)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14,
                     "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic"})
LW = 1.1


class Chart:
    def __init__(self, ax):
        self.ax, self.n = ax, {}

    def node(self, key, kind, x, y, w, h, text):
        ax = self.ax
        if kind == "term":
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle=f"round,pad=0,rounding_size={h / 2}", fc="white", ec="black", lw=LW))
        elif kind == "proc":
            ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="white", ec="black", lw=LW))
        elif kind == "io":
            s = 0.35
            ax.add_patch(Polygon([(x - w / 2 + s, y + h / 2), (x + w / 2 + s, y + h / 2),
                                  (x + w / 2 - s, y - h / 2), (x - w / 2 - s, y - h / 2)],
                                 closed=True, fc="white", ec="black", lw=LW))
        elif kind == "dec":
            ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                                 closed=True, fc="white", ec="black", lw=LW))
        ax.text(x, y, text, ha="center", va="center", fontsize=14, linespacing=1.15)
        self.n[key] = (x, y, w, h)

    def port(self, key, side):
        x, y, w, h = self.n[key]
        return {"t": (x, y + h / 2), "b": (x, y - h / 2), "l": (x - w / 2, y), "r": (x + w / 2, y)}[side]

    def line(self, pts, arrow=True, label=None, lpos=None, ha="left"):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color="black", lw=LW, solid_capstyle="butt")
        if arrow:
            self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                             arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22",
                                             lw=LW, color="black", shrinkA=0, shrinkB=0))
        if label:
            self.ax.text(*lpos, label, fontsize=14, ha=ha, va="center")


def main():
    fig, ax = plt.subplots(figsize=(6.5, 9.4))
    ax.set_xlim(0, 16.5); ax.set_ylim(3.6, 27.4); ax.axis("off")
    c = Chart(ax)
    X, XR, LOOP, BY = 5.0, 12.4, 15.9, 9.4
    c.node("start", "term", X, 26.7, 3.6, 0.9, "Start")
    c.node("in", "io", X, 25.2, 7.6, 1.3, "Frame $I_t$; $x_{t-1}$, $v_{t-1}$, $m$;\nhint (if available)")
    c.node("pred", "proc", X, 23.5, 7.6, 1.0, "Prediction $\\hat{x}_t$ by (1)")
    c.node("thr", "proc", X, 21.75, 7.6, 1.6, "Window $W_t$ around $\\hat{x}_t$;\nthreshold $\\theta_t$; silhouettes $b_i$")
    c.node("gate", "dec", X, 19.3, 7.2, 2.6, "Some same($b$)\nnot refuted by (2)?")
    c.node("upd", "proc", X, 16.65, 7.6, 2.0, "same($b$*) by (3);\n$x_t = c_{b*}$, $v_t$ by (1), $m = 0$;\nlabels of $b$* join the track")
    c.node("len", "dec", X, 13.9, 8.2, 2.8, "Track ended,\nseries ≥ 48 frames?")
    c.node("spec", "proc", X, 11.6, 7.6, 1.4, "Spectrum of the longest\nseries of $w_t$, $h_t$ (once)")
    c.node("ok", "dec", X, 9.2, 7.2, 2.6, "Flapping criterion\nmet?")
    c.node("out", "io", X, 7.0, 6.0, 1.0, "Fact $f_{flap}$")
    c.node("end", "term", X, 5.6, 3.6, 0.9, "End")

    c.node("miss", "proc", XR, 19.3, 5.0, 2.8, "unseen:\n$m = m + 1$;\n$x_t = \\hat{x}_t$, no label;\ngap in the series")
    c.node("lost", "dec", XR, 16.6, 4.2, 2.0, "$m > K$ ?")
    c.node("cue", "dec", XR, 13.9, 4.6, 2.2, "Hint\navailable?")
    c.node("reacq", "proc", XR, 11.5, 4.8, 1.4, "Re-acquisition\nnear the hint")
    c.node("wait", "io", XR, 9.2, 4.4, 1.4, "State\n\u201cunknown\u201d")

    v = lambda a, b: c.line([c.port(a, "b"), c.port(b, "t")])
    v("start", "in"); v("in", "pred"); v("pred", "thr"); v("thr", "gate")
    v("upd", "len"); v("spec", "ok"); v("out", "end"); v("miss", "lost")
    c.line([c.port("gate", "b"), c.port("upd", "t")], label="yes", lpos=(X + 0.2, 17.8))
    c.line([c.port("len", "b"), c.port("spec", "t")], label="yes", lpos=(X + 0.2, 12.5))
    c.line([c.port("ok", "b"), c.port("out", "t")], label="yes", lpos=(X + 0.2, 7.7))
    c.line([c.port("gate", "r"), c.port("miss", "l")], label="no", lpos=(9.3, 19.7), ha="center")
    c.line([c.port("lost", "b"), c.port("cue", "t")], label="yes", lpos=(XR + 0.2, 15.3))
    c.line([c.port("cue", "b"), c.port("reacq", "t")], label="yes", lpos=(XR + 0.2, 12.5))
    c.line([c.port("cue", "l"), (BY, 13.9), (BY, 9.2), c.port("wait", "l")],
           label="no", lpos=(BY + 0.25, 14.3), ha="left")
    c.line([(LOOP, 4.5), (LOOP, 26.15), (X + 1.6, 26.15)])
    ax.text(LOOP + 0.15, 21.5, "next frame $t + 1$", rotation=90, ha="left", va="center", fontsize=14)
    c.line([c.port("lost", "r"), (LOOP, 16.6)], arrow=False, label="no", lpos=(XR + 2.5, 16.95), ha="center")
    c.line([c.port("reacq", "r"), (LOOP, 11.5)], arrow=False)
    c.line([(c.port("wait", "r")[0] - 0.2, 9.2), (LOOP, 9.2)], arrow=False)
    c.line([c.port("len", "l"), (0.5, 13.9), (0.5, 4.5), (LOOP, 4.5)], arrow=False,
           label="no", lpos=(0.95, 14.3), ha="center")
    c.line([c.port("ok", "l"), (0.5, 9.2)], arrow=False, label="no", lpos=(0.95, 9.6), ha="center")
    out_dir = os.path.join(FIG, "en")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "fig_algo_en.png"), dpi=600, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
