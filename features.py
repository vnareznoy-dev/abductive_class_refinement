#!/usr/bin/env python3
"""Common observation pipeline: noisy grounding -> causal tracker -> 7 features.

Every class (real drone tracks, synthesised bird tracks) passes through THIS
function and nothing else, so the only thing that can separate the classes is
genuine kinematics -- not sampling artefacts or coordinate ranges. That is what
makes the comparison methodologically valid.

Grounding model (identical for all classes, same as the companion study):
  * position is observed with additive Gaussian error of std sigma (metres)
  * velocity is NOT given: it is estimated causally from an EMA-smoothed
    position sequence (backward difference, no look-ahead) -- a real tracker
  * heading and turn rate follow from the estimated velocity

Features consumed by air_objects.pl:
  f_speed       mean estimated speed                            (m/s)
  f_speed_cv    std/mean of estimated speed                     (-)
  f_hover       fraction of samples below the hover speed        (0..1)
  f_flap        wing-beat band oscillation present               (bool)
  f_turn_irreg  std of estimated heading rate                    (rad/s)
  f_straight    fraction of samples with near-zero heading rate  (0..1)
  f_turn_bias   MEAN SIGNED heading rate                         (rad/s)
"""
from __future__ import annotations

import numpy as np

EMA = 0.5            # causal position smoother (same constant as the companion study)
HOVER_SPEED = 1.0    # m/s   must match thr(hover_speed) in air_objects.pl
STRAIGHT_RATE = 0.05 # rad/s below this the object counts as flying straight
BEAT_LO, BEAT_HI = 2.0, 5.0   # Hz  wing-beat band (gull-sized birds)
# A white-noise signal already puts (BEAT_HI-BEAT_LO)/f_Nyquist of its power in
# the band (0.3 at 20 Hz sampling), so the decision threshold must sit clearly
# ABOVE that proportional share -- otherwise measurement noise alone reads as
# flapping. FLAP_MIN_RIPPLE additionally requires the oscillation to be
# physically meaningful: below it there is no beat to detect at all.
FLAP_RATIO = 0.45    # in-band power share that counts as a genuine wing beat
FLAP_MIN_RIPPLE = 0.02   # in-band amplitude as a fraction of mean speed
V_EMA = 0.3          # tracker smoothing of the VELOCITY estimate (see below)


MANOEUVRE_HI = 6.0   # Hz  above this, heading-rate power is noise, not flying
NOISE_SHARE_MAX = 0.50   # a quantity whose variance is mostly noise is not a fact


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _denoise_var(x: np.ndarray, dt: float, hi: float = MANOEUVRE_HI):
    """Variance of `x` with the broadband noise floor removed.

    The frequency principle used throughout this work, applied as an ESTIMATOR
    rather than as a gate. Genuine manoeuvring is band-limited -- no bird or
    multirotor changes heading at 6 Hz -- whereas grounding noise is broadband
    and flat. So the power density above `hi` is noise ONLY, and it estimates
    the noise floor across the whole band, which can then be subtracted. This
    needs no ground truth and no knowledge of sigma, so it is available to a
    real system.

    Returns (denoised variance, noise share of the raw variance).
    """
    n = len(x)
    if n < 32:
        return float(np.var(x)), 0.0
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    fr = np.fft.rfftfreq(n, d=dt)
    hf = fr >= hi
    pos = p[1:]
    if not hf.any() or pos.sum() <= 0:
        return float(np.var(x)), 0.0
    floor = float(np.median(p[hf]))            # per-bin noise power
    share = min(1.0, floor * len(pos) / float(pos.sum()))
    return float(np.var(x)) * (1.0 - share), share


def observe(pos: np.ndarray, dt: float, sigma: float, rng: np.random.Generator):
    """Noisy grounding + causal velocity estimation.

    Returns (v_raw, v_smooth): v_raw preserves the wing-beat band for spectral
    analysis; v_smooth is the tracker-filtered estimate used for heading.
    """
    obs = pos + (rng.normal(0.0, sigma, pos.shape) if sigma > 0 else 0.0)
    sm = np.empty_like(obs)
    sm[0] = obs[0]
    for k in range(1, len(obs)):
        sm[k] = EMA * obs[k] + (1.0 - EMA) * sm[k - 1]
    v = np.zeros_like(sm)
    v[1:] = (sm[1:] - sm[:-1]) / dt          # backward difference: causal
    v[0] = v[1]
    # TWO SIGNAL PATHS -- a single smoothing constant cannot serve both features:
    #   * heading RATE is one derivative beyond velocity, so noise compounds
    #     twice and a real tracker must low-pass the velocity before using it;
    #   * the wing beat lives at 2-5 Hz, exactly the band such a low-pass
    #     removes, so beat detection must run on the UNsmoothed estimate.
    # Operational systems separate these the same way: track filtering for
    # kinematics, spectral analysis of the raw stream for micro-motion.
    vs = np.empty_like(v)
    vs[0] = v[0]
    for k in range(1, len(v)):
        vs[k] = V_EMA * v[k] + (1.0 - V_EMA) * vs[k - 1]
    # Position noise, MEASURED from the data (sigma is unknown to a real
    # estimator). Estimated from the SECOND DIFFERENCE, not from the smoother
    # residual: obs - sm contains the EMA's tracking LAG, which for a moving
    # object is ~one step of travel and swamps the noise, so that estimator
    # rejects precisely the well-conditioned tracks. The second difference
    # cancels any constant-velocity component exactly, and white noise enters it
    # with variance 6 sigma^2.
    if sigma > 0 and len(obs) > 3:
        d2 = np.diff(obs[:, :2], n=2, axis=0)
        jitter = float(np.std(d2) / np.sqrt(6.0))
    else:
        jitter = 0.0
    return v, vs, jitter


def _flap_present(speed: np.ndarray, vz: np.ndarray, dt: float) -> bool:
    """Detect periodic energy in the wing-beat band of speed / vertical rate."""
    n = len(speed)
    if n < 32:
        return False
    freqs = np.fft.rfftfreq(n, d=dt)
    band = (freqs >= BEAT_LO) & (freqs <= BEAT_HI)
    if not band.any():
        return False
    mean_sp = float(np.mean(speed))
    if mean_sp <= 1e-6:
        return False
    ok = []
    for sig in (speed, vz):
        x = sig - sig.mean()
        if np.allclose(x, 0):
            continue
        p = np.abs(np.fft.rfft(x)) ** 2
        tot = p[1:].sum()
        if tot <= 0:
            continue
        share = p[band].sum() / tot
        # in-band amplitude, expressed relative to mean speed
        amp = np.sqrt(2.0 * p[band].sum()) / n / mean_sp
        ok.append(share > FLAP_RATIO and amp > FLAP_MIN_RIPPLE)
    return any(ok)


HEADING_SNR = 3.0     # per-step displacement must exceed this * displacement noise


def extract(pos: np.ndarray, dt: float, sigma: float,
            rng: np.random.Generator) -> dict:
    """pos[N,3] ground truth -> feature dict computed from NOISY observation.

    A feature is returned as None when its ESTIMATOR IS UNDEFINED, not merely
    noisy. An unstated fact activates no integrity constraint and satisfies no
    signature, which is the formalism's native representation of "not
    measurable" -- far safer than asserting a degenerate number.

    Heading is the case in point. psi = atan2(vy, vx) is undefined as |v| -> 0,
    so for a STATION-KEEPING object every heading-derived quantity is numerical
    noise regardless of grounding quality: measured on hovering drones at
    sigma = 0, turn_irreg came out at 6.7 rad/s and speed_cv at 0.51, against
    authored thresholds of 0.35 and 0.25. Those are not noisy readings of a
    real quantity, they are readings of a quantity that does not exist. The
    guard below is derived from the MEASURED jitter, so it needs no knowledge
    of sigma.
    """
    v_raw, v_sm, jitter = observe(pos, dt, sigma, rng)
    # speed / hover and the wing beat come from the raw estimate, so genuine
    # airspeed variation is preserved
    sp = np.linalg.norm(v_raw[:, :2], axis=1)
    sp_sm = np.linalg.norm(v_sm[:, :2], axis=1)
    # heading and its rate come from the tracker-filtered estimate
    psi = np.arctan2(v_sm[:, 1], v_sm[:, 0])
    rate = _wrap(np.diff(psi)) / dt if len(psi) > 1 else np.zeros(1)

    mean_sp = float(sp.mean())
    # Is heading defined? Compare the per-step displacement actually travelled
    # against the noise on that displacement (jitter on each of two samples).
    # Two independent legs, both of which must hold:
    #   (a) MEASUREMENT: the step travelled must stand out of the jitter;
    #   (b) PHYSICAL: an object holding position has no meaningful heading at
    #       all, however perfectly it is measured. At sigma = 0 leg (a) passes
    #       trivially (jitter = 0) while the heading of a station-keeping drone
    #       is still pure churn, so leg (b) is what catches that case. It reuses
    #       the authored hover speed rather than introducing a new constant.
    # Leg (a) must test the noise on the quantity ACTUALLY USED. Heading is not
    # taken from a single backward difference but from the EMA-smoothed velocity
    # `v_sm`, whose equivalent averaging length for coefficient alpha is
    # (2 - alpha)/alpha samples. The relevant travel is therefore the
    # displacement over THAT window, while the endpoint noise on it does not
    # grow with the window. Comparing a single step instead understates the
    # available signal by that factor.
    n_eff = (2.0 - V_EMA) / V_EMA
    med_sp = float(np.median(sp_sm))
    travel = med_sp * dt * n_eff
    disp_noise = jitter * np.sqrt(2.0)
    heading_ok = (travel > HEADING_SNR * disp_noise) and (med_sp > HOVER_SPEED)

    # The heading guard licenses HEADING. Its rate is one derivative further
    # out, so its own noise must be removed separately: at sigma = 0 a soaring
    # bird showed 0.032 rad/s, at sigma = 0.15 the same track read above the
    # 0.35 threshold purely from noise inflation.
    rate_var, rate_noise_share = _denoise_var(rate, dt)
    sp_var, sp_noise_share = _denoise_var(sp, dt)

    # Subtracting the noise floor is not enough on its own: the floor is
    # estimated, and removing 90 % of a variance 30 dB above the signal still
    # leaves a residual that grows with sigma (measured: 0.20 -> 0.47 -> 0.94
    # rad/s for sigma 0.05 -> 0.15 -> 0.30). A signal that far below the noise
    # is not recoverable by any estimator, so the measured noise share is used
    # as the GROUNDABILITY TEST rather than as a correction. Being measurable
    # without ground truth, it is available to a real system.
    rate_ok = heading_ok and rate_noise_share <= NOISE_SHARE_MAX
    sp_cv_ok = heading_ok and sp_noise_share <= NOISE_SHARE_MAX

    out = {
        "f_speed": mean_sp,
        "f_hover": float((sp < HOVER_SPEED).mean()),
        "f_flap": _flap_present(sp, v_raw[:, 2], dt),
        # cv = std/mean is degenerate as mean -> 0; gate on the heading test too
        "f_speed_cv": (float(np.sqrt(sp_var) / mean_sp)
                       if sp_cv_ok and mean_sp > 1e-6 else None),
        "f_turn_irreg": float(np.sqrt(rate_var)) if rate_ok else None,
        "f_straight": float((np.abs(rate) < STRAIGHT_RATE).mean()) if rate_ok else None,
        "f_turn_bias": float(rate.mean()) if rate_ok else None,
    }
    out["_heading_ok"] = heading_ok
    out["_rate_ok"] = rate_ok
    out["_rate_noise_share"] = rate_noise_share
    return out


if __name__ == "__main__":
    import birds
    rng = np.random.default_rng(0)
    print(f"{'mode':10s} {'sigma':>6s} {'speed':>6s} {'cv':>5s} {'hover':>6s} "
          f"{'flap':>5s} {'irreg':>6s} {'strt':>5s} {'bias':>6s}")
    for mode in birds.MODES:
        tr = [x for x in birds.make_bird_set(1, 200, seed=3) if x["mode"] == mode][0]
        for sg in (0.0, 0.3):
            f = extract(tr["pos"], birds.DT, sg, np.random.default_rng(1))
            g = lambda k, w, p: ("—".rjust(w) if f[k] is None
                                 else f"{f[k]:{w}.{p}f}")
            print(f"{mode:10s} {sg:6.2f} {f['f_speed']:6.2f} {g('f_speed_cv',5,2)} "
                  f"{f['f_hover']:6.2f} {str(f['f_flap']):>5s} {g('f_turn_irreg',6,3)} "
                  f"{g('f_straight',5,2)} {g('f_turn_bias',6,3)}")
