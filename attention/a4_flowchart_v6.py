#!/usr/bin/env python3
"""Algorithm flowchart of the attention holding (ISO 5807 / ДСТУ 19.701 symbols):
terminator, input/output (parallelogram), process (rectangle), decision (diamond).
Ukrainian labels, Times New Roman 14 pt, 300 dpi -> fig/fig_algo.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "fig")
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
    c.node("start", "term", X, 26.7, 3.6, 0.9, "Початок")
    c.node("in", "io", X, 25.2, 7.6, 1.3, "Кадр $I_t$; стан $x_{t-1}$, $v_{t-1}$, $m$;\nпідказка (за наявності)")
    c.node("pred", "proc", X, 23.5, 7.6, 1.0, "Прогноз $\\hat{x}_t = x_{t-1} + v_{t-1}$")
    c.node("thr", "proc", X, 21.8, 7.6, 1.4, "Вікно $W_t$ навколо $\\hat{x}_t$;\nпоріг $\\theta_t$ за (2), (3)")
    c.node("gate", "dec", X, 19.3, 7.2, 2.6, "Силует\nне далі $L_t$ від прогнозу?")
    c.node("upd", "proc", X, 16.8, 7.6, 1.4, "$x_t = c_b$, $v_t$ за (4), $m = 0$;\nдодати $(w_t, h_t)$ до ряду")
    c.node("len", "dec", X, 14.3, 7.4, 2.8, "Ряд ≥ 48 кадрів\nбез пропусків?")
    c.node("spec", "proc", X, 11.9, 7.6, 1.4, "Спектр останніх 48 кадрів\nряду $w_t$, $h_t$")
    c.node("ok", "dec", X, 9.4, 7.2, 2.6, "Критерій (6)\nвиконано?")
    c.node("out", "io", X, 7.1, 6.0, 1.0, "Факт для\nALP [13]")
    c.node("end", "term", X, 5.6, 3.6, 0.9, "Кінець")

    c.node("miss", "proc", XR, 19.3, 4.8, 2.4, "$m = m + 1$;\nінерція $x_t = \\hat{x}_t$;\nпропуск у ряді")
    c.node("lost", "dec", XR, 16.6, 4.2, 2.0, "$m > K$ ?")
    c.node("cue", "dec", XR, 13.9, 4.6, 2.2, "Є підказка?")
    c.node("reacq", "proc", XR, 11.5, 4.8, 1.4, "Перезахоплення\nбіля підказки")
    c.node("wait", "io", XR, 9.2, 4.4, 1.4, "Стан\n«невідомо»")

    v = lambda a, b: c.line([c.port(a, "b"), c.port(b, "t")])
    v("start", "in"); v("in", "pred"); v("pred", "thr"); v("thr", "gate")
    v("upd", "len"); v("spec", "ok"); v("out", "end"); v("miss", "lost")
    c.line([c.port("gate", "b"), c.port("upd", "t")], label="так", lpos=(X + 0.2, 17.75))
    c.line([c.port("len", "b"), c.port("spec", "t")], label="так", lpos=(X + 0.2, 12.85))
    c.line([c.port("ok", "b"), c.port("out", "t")], label="так", lpos=(X + 0.2, 7.85))
    c.line([c.port("gate", "r"), c.port("miss", "l")], label="ні", lpos=(9.3, 19.7), ha="center")
    c.line([c.port("lost", "b"), c.port("cue", "t")], label="так", lpos=(XR + 0.2, 15.3))
    c.line([c.port("cue", "b"), c.port("reacq", "t")], label="так", lpos=(XR + 0.2, 12.5))
    c.line([c.port("cue", "l"), (BY, 13.9), (BY, 9.2), c.port("wait", "l")],
           label="ні", lpos=(BY + 0.25, 14.3), ha="left")
    # повернення до наступного кадру
    c.line([(LOOP, 4.5), (LOOP, 26.15), (X + 1.6, 26.15)])
    ax.text(LOOP + 0.15, 21.5, "наступний кадр $t + 1$", rotation=90, ha="left", va="center", fontsize=14)
    c.line([c.port("lost", "r"), (LOOP, 16.6)], arrow=False, label="ні", lpos=(XR + 2.5, 16.95), ha="center")
    c.line([c.port("reacq", "r"), (LOOP, 11.5)], arrow=False)
    c.line([(c.port("wait", "r")[0] - 0.2, 9.2), (LOOP, 9.2)], arrow=False)
    c.line([c.port("len", "l"), (0.5, 14.3), (0.5, 4.5), (LOOP, 4.5)], arrow=False,
           label="ні", lpos=(0.95, 14.7), ha="center")
    c.line([c.port("ok", "l"), (0.5, 9.4)], arrow=False, label="ні", lpos=(0.95, 9.8), ha="center")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_algo_v6.png"), dpi=300, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
