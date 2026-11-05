#!/usr/bin/env python3
"""Figure for the AAIT article 4b: the core of the identity theory in Prolog, printed verbatim from
abductive_class_refinement/attention/a4_track_alp.pl (lines from the first ':- dynamic' to the end;
the header comment is omitted; padding before/inside end-of-line comments collapsed to two spaces). Courier New, 600 dpi -> fig/en/fig_prolog_en.png
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import A4   # noqa: E402  (works in article4/code and in the repository attention/)
PL = os.path.join(HERE, "a4_track_alp.pl")
if not os.path.exists(PL):
    PL = os.path.join(HERE, "..", "..", "abductive_class_refinement", "attention", "a4_track_alp.pl")
OUT = os.path.join(A4, "fig", "en", "fig_prolog_en.png")
CN = "/System/Library/Fonts/Supplemental/Courier New.ttf"
font_manager.fontManager.addfont(CN)


def main():
    lines = open(PL, encoding="utf-8").read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.startswith(":- dynamic"))
    code = [re.sub(r"\s{3,}", "  ", l.rstrip()) if "%" in l else l.rstrip() for l in lines[i0:]]   # aligned comment padding collapsed
    while code and not code[-1].strip(): code.pop()
    fig = plt.figure(figsize=(6.9, 0.165 * len(code) + 0.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, len(code) + 0.6)
    for k, l in enumerate(code):
        col = "#555555" if l.lstrip().startswith("%") or "%" in l and l.index("%") > 0 and False else "black"
        ax.text(0.01, len(code) - k, l, family="Courier New", fontsize=8.6, va="center", ha="left", color=col)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=600)
    print("saved", OUT, len(code), "lines, longest", max(map(len, code)))


if __name__ == "__main__":
    main()
