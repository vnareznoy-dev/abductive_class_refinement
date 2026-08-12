#!/usr/bin/env python3
"""Physically parameterised BIRD trajectory generator (declared synthetic).

Data-availability note for the paper: real bird tracks at the sampling rate this
study needs (>= 10 Hz, set by the wing-beat band) are not openly available. The
public GPS collections that do exist -- e.g. the INBO/LifeWatch herring-gull
archive -- sample at 1-5 min, three orders of magnitude too coarse to observe
wing beats or turn dynamics. Bird motion is therefore SYNTHESISED from published
flight parameters; this is declared as a limitation, and drones remain real.

Flight parameters (medium gull-sized bird, level flight):
  cruise speed        8-15 m/s
  wing-beat frequency 2.5-3.5 Hz   -> drives speed/vertical micro-oscillation
  beat speed ripple   5-12 % of cruise
  thermal circling    radius 20-60 m, one-way turn, low irregularity
  manoeuvre turn rate up to ~0.8 rad/s, sign-changing

Three behaviour modes are generated, matching the theory's bird signatures:
  flapping   - beating flight with irregular course changes
  soaring    - beat-free circling in a thermal (one-way turn)
  gliding    - beat-free straight glide  (deliberately unexplainable: the
               theory abstains on it; kept so the limitation is measurable)
"""
from __future__ import annotations

import numpy as np

DT = 0.04           # s  -> 25 Hz, matches CTU-MRS stride=2; above the >=10 Hz floor
MODES = ("flapping", "soaring", "gliding")


def _unit(a):
    return np.array([np.cos(a), np.sin(a)])


def make_bird(mode: str, steps: int, rng: np.random.Generator):
    """Return (t[N], pos[N,3]) for one bird track in a local ENU frame (metres)."""
    v0 = rng.uniform(8.0, 15.0)                 # cruise speed
    f_beat = rng.uniform(2.5, 3.5)              # wing-beat frequency, Hz
    ripple = rng.uniform(0.05, 0.12)            # speed modulation depth
    psi = rng.uniform(0, 2 * np.pi)             # heading
    p = np.array([rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(40, 120)])

    if mode == "soaring":
        radius = rng.uniform(20.0, 60.0)
        omega = (v0 / radius) * rng.choice([-1.0, 1.0])   # one-way circling
        climb = rng.uniform(0.5, 1.5)                     # thermal lift, m/s
    elif mode == "gliding":
        omega = 0.0
        climb = rng.uniform(-1.0, -0.3)                   # slow sink
    else:                                                 # flapping
        omega = 0.0
        climb = 0.0

    t = np.arange(steps) * DT
    pos = np.zeros((steps, 3))
    for k in range(steps):
        # wing beats modulate airspeed and produce vertical micro-oscillation
        if mode == "flapping":
            beat = np.sin(2 * np.pi * f_beat * t[k])
            v = v0 * (1.0 + ripple * beat)
            vz = 0.6 * ripple * v0 * np.cos(2 * np.pi * f_beat * t[k])
            # irregular, sign-changing course corrections
            psi += rng.normal(0.0, 0.55) * DT
        else:
            v = v0
            vz = climb
            psi += omega * DT
            if mode == "soaring":                          # slight roughness
                psi += rng.normal(0.0, 0.04) * DT

        pos[k] = p
        p = p + np.concatenate([_unit(psi) * v, [vz]]) * DT
    return t, pos


def make_bird_set(n_per_mode: int, steps: int, seed: int):
    """Balanced set over the three behaviour modes."""
    rng = np.random.default_rng(seed)
    out = []
    for mode in MODES:
        for _ in range(n_per_mode):
            tt, pp = make_bird(mode, steps, rng)
            out.append({"class": "bird", "mode": mode, "t": tt, "pos": pp})
    return out


if __name__ == "__main__":
    s = make_bird_set(4, 200, seed=1)
    print(f"{len(s)} bird tracks, {len(s[0]['t'])} samples @ {1/DT:.0f} Hz")
    for mode in MODES:
        tr = [x for x in s if x["mode"] == mode][0]
        d = np.linalg.norm(np.diff(tr["pos"], axis=0), axis=1) / DT
        print(f"  {mode:9s} speed {d.mean():5.2f} m/s (sd {d.std():4.2f}), "
              f"alt {tr['pos'][0,2]:5.1f} -> {tr['pos'][-1,2]:5.1f} m")
