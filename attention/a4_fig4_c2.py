#!/usr/bin/env python3
"""Fig. 4 of 4b for the main variant 2 L_t (revision): the same bird track figure as a4_attention_fig.py (UA) and
a4_attention_fig_en.py (EN), run with the Python twin limited by 2 L_t (A4_ALP_MULT=2). The original drawing code
is reused unchanged; only the output names get the suffix _c2 (the c = 1 figures stay as they are).
Outputs: fig/fig_attn_bird_alp_c2.png, fig/en/fig_attn_bird_alp_c2_en.png"""
import os
import sys

os.environ["A4_ALP_MULT"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_attention_fig as UA        # noqa: E402
import a4_attention_fig_en as EN     # noqa: E402

for mod in (UA, EN):
    orig = mod.draw

    def draw(log, fps, out, *a, _o=orig, **k):
        return _o(log, fps, out.replace("_alp", "_alp_c2"), *a, **k)
    mod.draw = draw
    print("====", mod.__name__); mod.bird(alp=True)
