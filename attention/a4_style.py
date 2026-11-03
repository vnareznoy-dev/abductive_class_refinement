"""Shared figure style of the articles (after Bauersfeld et al., NeuroBEM, RSS 2021):
full axes frame, light grid, thin coloured lines, one boxed legend above the panels,
colour-blind safe palette (Okabe & Ito). Journal constraints are kept: Times New Roman
14 pt, Ukrainian labels, decimal comma, units after a comma."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
font_manager.fontManager.addfont(TNR)

BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
SKY, YELLOW, GREY, BLACK = "#56B4E9", "#E69F00", "#8C8C8C", "#000000"
# fixed colour per strategy, used in every figure
C = {"tbd": GREY, "bytetrack": SKY, "botsort": PURPLE, "att_self": YELLOW,
     "att_none": BLUE, "att_ext1Hz": GREEN, "truth": BLACK, "ours": ORANGE}
FILL_HOLD, FILL_DEC = "#DCEBF7", "#FBE3D3"     # panels of the scheme: holding / decision
W_IN = 6.5                                      # 16.5 cm text width

RC = ({
    "font.family": "Times New Roman", "font.size": 14,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "axes.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#D9D9D9", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.linewidth": 1.3, "patch.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "legend.fontsize": 14, "legend.borderpad": 0.4,
    "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
    "savefig.dpi": 300,
})


def apply():
    plt.rcParams.update(RC)


apply()


def comma(ax, x=True, y=True):
    f = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
    if x:
        ax.xaxis.set_major_formatter(f)
    if y:
        ax.yaxis.set_major_formatter(f)


def top_legend(fig, handles, labels, ncol, y=1.0):
    """One boxed legend centred above all panels."""
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol)


def panel(ax, letter):
    ax.set_title(letter, loc="left", fontsize=14, pad=4)
