#!/usr/bin/env python3
"""Article 3 v22, Fig. 5 (reviewer remark 16): silhouette spectra of a flapping bird
next to the spectra of non-flapping UAVs, with the same normalisation.

Also re-runs the specificity test with the SAME four silhouette measures that are
tested on the bird (width, height, area, h/w), because the published
flap_specificity.py tests only three on UAVs (height, area, h/w).

Run:  python3 fig5_spectra.py
Writes reports/fig5_spectra.png and reports/fig5_spectra.txt
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ACR)
import flap_real as FR            # noqa: E402
import flap_specificity as FS     # noqa: E402

OUT_FIG = os.path.join(ACR, "reports", "fig5_spectra.png")
OUT_TXT = os.path.join(ACR, "reports", "fig5_spectra.txt")
LO, HI = 1.5, 6.0
WIN = 48


def uk(t):
    return t.replace(".", ",")


def spectrum(sig, fps):
    """Detrended power spectrum, normalised to the total power (DC excluded)."""
    x = np.asarray(sig, float)
    x = x - x.mean()
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    p = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), d=1.0 / fps)
    return fr[1:], p[1:] / p[1:].sum()


def main():
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    log = []

    # bird: verified cormorant run (frames 1351-1398, 48 frames, 25 Hz)
    seq, fps_b = FR.track(FR.VID, 1240, 1800)
    run = FR.runs(seq)[0]
    a = np.array([(s[3], s[4]) for s in run], float)
    wb, hb = a[:, 0], a[:, 1]
    fb, pb = spectrum(wb, fps_b)
    tb = FR.band_test(wb, fps_b)
    log.append(f"bird (cormorant) width: n={len(wb)} share={tb['share']:.2f} "
               f"floor={tb['base']:.2f} ratio={tb['share']/tb['base']:.2f} peak={tb['peak']:.2f} Hz")

    # UAVs: 9 continuous annotated runs, 25 Hz (assumed frame rate, as in flap_specificity.py)
    runs = FS.collect_drone()
    fps_d = FS.FPS
    fires, total = 0, 0
    per_run = []
    win_spectra = []
    for i, r in enumerate(runs):
        w, h = r[:, 0], r[:, 1]
        sigs = {"width": w, "height": h, "area": w * h, "h/w": h / np.maximum(w, 1e-6)}
        cells = []
        for name, sig in sigs.items():
            t = FS.band_test(sig)
            total += 1
            fires += int(bool(t and t["fired"]))
            cells.append(f"{name} {t['share']:.2f}{'*' if t['fired'] else ''}" if t else f"{name} -")
        per_run.append(f"run {i+1}: {len(r)} frames | " + " | ".join(cells))
        # same window length as the bird: every 48-frame window, stride 24, width series
        for s in range(0, len(w) - WIN + 1, 24):
            win_spectra.append(spectrum(w[s:s + WIN], fps_d)[1])
    log += per_run
    log.append(f"specificity with 4 measures (incl. width): {fires} of {total} signals fired")
    W = np.array(win_spectra)
    n_long = sum(1 for r in runs if len(r) >= WIN)
    fd = spectrum(np.arange(WIN, dtype=float) ** 0 + np.random.default_rng(0).normal(size=WIN), fps_d)[0]
    band = (fd >= LO) & (fd <= HI)
    shares = W[:, band].sum(1)
    floor = (HI - LO) / (fps_d / 2)
    log.append(f"UAV 48-frame width windows: n={len(W)}; band share median {np.median(shares):.2f}, "
               f"5-95 % {np.percentile(shares,5):.2f}-{np.percentile(shares,95):.2f}; "
               f"windows above 2x floor ({2*floor:.2f}): {(shares > 2*floor).sum()}")

    # ---- figure (Ukrainian labels) ----
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for k, (f, title) in enumerate(((fb, "а) птах (баклан), 48 кадрів, 25 Гц"),
                                    (fd, f"б) апарати: {len(W)} вікон по 48 кадрів ({n_long} прогонів)"))):
        ax[k].axvspan(LO, HI, color="#f6d7b8", alpha=0.6, lw=0, label="смуга махів 1,5–6 Гц")
        ax[k].set_title(title)
        ax[k].set_xlabel("частота, Гц")
        ax[k].grid(alpha=0.35)
        ax[k].set_xlim(0, 12.5)
    nb = len(fb)
    ax[0].plot(fb, pb, color="#1f5fa8", lw=2.0, label="ряд розмаху силуету")
    ax[0].axhline(1.0 / nb, color="0.35", ls="--", lw=1.2, label="рівень білого шуму")
    ax[0].set_ylabel("частка потужності")
    lo_q, hi_q = np.percentile(W, 5, axis=0), np.percentile(W, 95, axis=0)
    ax[1].fill_between(fd, lo_q, hi_q, color="0.75", alpha=0.7, label="5–95 % вікон")
    ax[1].plot(fd, np.median(W, axis=0), color="0.15", lw=2.0, label="медіана вікон")
    ax[1].axhline(1.0 / len(fd), color="0.35", ls="--", lw=1.2, label="рівень білого шуму")
    ax[0].text(0.98, 0.95, uk(f"частка в смузі {tb['share']:.2f}\n= {tb['share']/tb['base']:.1f} × рівня шуму {tb['base']:.2f}"),
               transform=ax[0].transAxes, ha="right", va="top", fontsize=10,
               bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    ax[1].text(0.98, 0.95, uk(f"частка в смузі: медіана {np.median(shares):.2f}\n(рівень шуму {floor:.2f})"),
               transform=ax[1].transAxes, ha="right", va="top", fontsize=10,
               bbox=dict(boxstyle="round", fc="white", ec="0.6"))
    for k in range(2):
        h_, l_ = ax[k].get_legend_handles_labels()
        ax[k].legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=True)
    from matplotlib.ticker import FuncFormatter
    for k in range(2):
        ax[k].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=220, bbox_inches="tight")
    open(OUT_TXT, "w").write("\n".join(log) + "\n")
    print("\n".join(log))
    print("saved", OUT_FIG)


if __name__ == "__main__":
    main()
