#!/usr/bin/env python3
"""Unit-test the abductive class theory on hand-built cases (no data needed)."""
from __future__ import annotations
import os, sys
from pyswip import Prolog

PL = os.path.join(os.path.dirname(__file__), "air_objects.pl")
FACTS = ("f_det(_,_)", "f_speed(_)", "f_speed_cv(_)", "f_hover(_)",
         "f_flap(_)", "f_turn_irreg(_)", "f_straight(_)", "f_turn_bias(_)")


class Theory:
    def __init__(self):
        self.p = Prolog()
        self.p.consult(PL)

    def clear(self):
        for f in FACTS:
            list(self.p.query(f"retractall({f})"))

    def classify(self, det=None, speed=8.0, cv=0.10, hover=0.0,
                 flap=False, turn=0.10, straight=0.80, bias=0.0):
        self.clear()
        if det is not None:
            self.p.assertz(f"f_det({det[0]},{det[1]})")
        self.p.assertz(f"f_speed({speed})")
        self.p.assertz(f"f_speed_cv({cv})")
        self.p.assertz(f"f_hover({hover})")
        self.p.assertz(f"f_flap({'true' if flap else 'false'})")
        self.p.assertz(f"f_turn_irreg({turn})")
        self.p.assertz(f"f_straight({straight})")
        self.p.assertz(f"f_turn_bias({bias})")
        r = list(self.p.query("classify(C, W)"))
        if not r:
            return "NO_SOLUTION", "-"
        return str(r[0]["C"]), str(r[0]["W"])


CASES = [
    # (name, kwargs, expected class)
    ("дрон зависає, детектор каже drone",
     dict(det=("drone", 0.9), speed=0.4, hover=0.5, straight=0.1, turn=0.05), "drone"),
    ("дрон летить рівно, детектор каже drone",
     dict(det=("drone", 0.9), speed=12.0, straight=0.9, turn=0.05), "drone"),
    ("*** детектор каже drone, але махові коливання -> РЕВІЗІЯ",
     dict(det=("drone", 0.95), speed=11.0, cv=0.35, flap=True, hover=0.0,
          turn=0.5, straight=0.2), "bird"),
    ("45 м/с тепер У КОНВЕРТІ дрона (порог 125) -> підтвердження",
     dict(det=("drone", 0.9), speed=45.0, straight=0.95, turn=0.02), "drone"),
    ("140 м/с -> поза конвертом дрона й птаха",
     dict(det=("drone", 0.9), speed=140.0, straight=0.98, turn=0.02), "unknown"),
    ("птах: махи + нерегулярні розвороти, детектор каже bird",
     dict(det=("bird", 0.8), speed=10.0, cv=0.3, flap=True, turn=0.5,
          straight=0.2), "bird"),
    ("*** детектор каже bird, але тривале зависання -> РЕВІЗІЯ",
     dict(det=("bird", 0.9), speed=0.3, hover=0.6, turn=0.05, straight=0.1), "drone"),
    ("немає довіри до детектора -> вирішує кінематика (махи)",
     dict(det=("drone", 0.2), speed=10.0, cv=0.3, flap=True, turn=0.5,
          straight=0.2), "bird"),
    ("детектора немає -> кінематика (зависання)",
     dict(det=None, speed=0.4, hover=0.5, turn=0.05, straight=0.1), "drone"),
    ("літак як контроль: 200 м/с, прямо -> поза обома конвертами",
     dict(det=("drone", 0.9), speed=200.0, straight=0.99, turn=0.01), "unknown"),
    # --- нові випадки після перегляду порогів ---
    ("швидкий дрон 100 м/с (у новому конверті), детектор каже drone",
     dict(det=("drone", 0.9), speed=100.0, straight=0.95, turn=0.03), "drone"),
    ("ПАРЯЧИЙ птах у термику: кружить рівно, без махів -> ПІДТВЕРДЖЕНО bird",
     dict(det=("bird", 0.85), speed=12.0, cv=0.10, flap=False, hover=0.0,
          turn=0.08, straight=0.20, bias=0.25), "bird"),
    ("планер без кружіння: кінематика МОВЧИТЬ -> мітка детектора лишається",
     dict(det=("bird", 0.85), speed=12.0, cv=0.10, flap=False, hover=0.0,
          turn=0.08, straight=0.85, bias=0.0), "bird"),
    ("КРУЖНИЙ ДРОН (барражування) -> кружіння НЕ ревізує drone",
     dict(det=("drone", 0.9), speed=9.0, cv=0.08, flap=False, hover=0.0,
          turn=0.10, straight=0.20, bias=0.30), "drone"),
    ("махи мають приоритет над парінням",
     dict(det=("bird", 0.8), speed=10.0, cv=0.3, flap=True, turn=0.5,
          straight=0.2, bias=0.30), "bird"),
    ("птах 30 м/с -> поза конвертом птаха, детектор каже bird",
     dict(det=("bird", 0.9), speed=30.0, straight=0.9, turn=0.05), "drone"),
]


def main():
    t = Theory()
    ok = bad = 0
    for name, kw, exp in CASES:
        c, w = t.classify(**kw)
        good = (c == exp)
        ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
        print(f"  {'OK ' if good else '!! '} {name}")
        print(f"       -> {c}   ({w})")
        if not good:
            print(f"       ОЧІКУВАНО: {exp}")
    print(f"\n{ok} пройшло, {bad} не пройшло")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
