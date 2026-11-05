#!/usr/bin/env python3
"""English versions of the three figures of the identity article (4б) for AAIT.
Copies a4_identity_real.py, a4_attention_fig.py and a4_sens.py into *_en.py with Ukrainian
string literals translated, decimal comma -> decimal point (AAIT rule), panel letters a/b/c,
outputs in fig/en/. The data and the drawing code are unchanged. Run: python3 make_en_figs.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = [('"а"', '"a"'), ('"б"', '"b"'), ('"в"', '"c"'), ('.replace(".", ",")', '')]
TR = {
    "a4_identity_real.py": [
        ('"а) кадр треку птаха (відеозапис атолу Мідвей)"', '"a) frame of a bird track (Midway Atoll video)"'),
        ('"б) гіпотези тотожності"', '"b) identity hypotheses"'),
        ('"unseen (ціль невидима)"', '"unseen (target not visible)"'),
        ('"обмеження цілісності:\\n"', '"integrity constraint:\\n"'),
        ('" пікс."', '" px"'), ('", допустима"', '", admissible"'), ('" і "', '" and "'),
        ('", допустимі"', '", admissible"'), ('", спростована"', '", refuted"'),
        ('"мінімальне пояснення:\\nнайменше відхилення "', '"minimal explanation:\\nsmallest deviation "'),
        ('r"$\\rightarrow$ ціль $= b_1$"', 'r"$\\rightarrow$ target $= b_1$"'),
        ('r"мітки лише від $b_1$ $\\rightarrow$"', 'r"labels only from $b_1$ $\\rightarrow$"'),
        ('"\\nабдуктивне уточнення класу"', '"\\nabductive class refinement"'),
        ('"якщо всі same(b)\\nспростовано –\\nабдукується unseen:\\nкадр без мітки"', '"if all same(b)\\nare refuted,\\nunseen is abduced:\\nframe without a label"'),
    ],
    "a4_attention_fig.py": [
        ('"утримання"', '"holding"'), ('"інерція"', '"extrapolation"'), ('"перезахоплення"', '"re-acquisition"'), ('"втрата"', '"lost"'),
        ('f"кадр {i}: {ST_NAME[e[\'state\']]}"', 'f"frame {i}: {ST_NAME[e[\'state\']]}"'),
        ('f"класифікатор c = {e[\'conf\']:.2f}"', 'f"classifier c = {e[\'conf\']:.2f}"'),
        ('f"кадр {i}\\n{ST_NAME[e[\'state\']]}\\n$c_t$ = {e[\'conf\']:.2f}"', 'f"frame {i}\\n{ST_NAME[e[\'state\']]}\\n$c_t$ = {e[\'conf\']:.2f}"'),
        ('"поріг 0,25"', '"threshold 0.25"'), ('"найдовший ряд"', '"longest run"'), ('", пкс"', '", px"'),
        ('"Номер кадру"', '"Frame number"'), ('"Частота, Гц"', '"Frequency, Hz"'), ('"Частка\\nпотужності"', '"Power\\nshare"'),
        ('"виконано"', '"met"'), ('"не виконано"', '"not met"'),
        ('f"критерій {verdict}: {decision_word if r[\'fired\'] else \'«невідомо»\'}"', 'f"criterion {verdict}: {decision_word if r[\'fired\'] else \'“unknown”\'}"'),
        ('"«птах»"', '"“bird”"'),
        ('f"fig_attn_bird{sfx}.png"', 'f"en/fig_attn_bird{sfx}_en.png"'), ('f"fig_attn_uav{sfx}.png"', 'f"en/fig_attn_uav{sfx}_en.png"'),
        ('    write_gif(log, 200, os.path.join(A4, "fig", "supp", f"attn_bird{sfx}.gif"), X.FPS_MIDWAY)', '    pass  # no GIF for the English figure'),
    ],
    "a4_sens.py": [
        ('"правильно"', '"correct"'), ('"хибно"', '"wrong"'), ('"БпЛА → «птах»"', '"UAV → “bird”"'),
        ('"Поріг довіри до мітки"', '"Label trust threshold"'), ('"Частка цілей, %"', '"Share of targets, %"'),
        ('"класифікатор без ALP"', '"classifier without ALP"'),
        ('os.path.join(A4, "fig", "fig_sens.png")', 'os.path.join(A4, "fig", "en", "fig_sens_en.png")'),
    ],
}


def main():
    os.makedirs(os.path.join(HERE, "..", "fig", "en"), exist_ok=True)
    for src, pairs in TR.items():
        s = open(os.path.join(HERE, src), encoding="utf-8").read()
        for a, b in pairs + COMMON:
            if a in s:
                s = s.replace(a, b)
            elif (a, b) not in COMMON:
                sys.exit(f"{src}: not found: {a}")
        s = re.sub(r"\n\s*ST\.comma\([^)]*\)", "", s)                 # no decimal-comma formatter
        s = s.replace("ST.comma(ax)", "None").replace("comma(a_)", "None")
        left = [m for m in re.findall(r'"[^"\n]*[а-яіїєґА-ЯІЇЄҐ][^"\n]*"', s.split('"""', 2)[-1])]
        if left:
            print(src, "untranslated:", left)
        dst = src.replace(".py", "_en.py")
        open(os.path.join(HERE, dst), "w", encoding="utf-8").write(s)
        print("wrote", dst)


if __name__ == "__main__":
    main()
