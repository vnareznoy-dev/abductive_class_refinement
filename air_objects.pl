% air_objects.pl
% ---------------------------------------------------------------------------
% Abductive determination of the class of an airborne object from TWO
% independent evidence channels: the appearance evidence of a neural detector
% and the kinematic evidence of the observed track.
%
% Classes: drone | bird | unknown (fallback).
%
% Formal statement (same criterion as the intent model of the companion work):
%
%     Delta* = argmin |Delta|  s.t.  T u Delta |= O
%                                     T u Delta u IC  not|- false
%
% Delta is a single class assumption (|Delta| = 1), so minimality reduces to:
% among the classes whose signature is NOT refuted by the kinematic
% observation, prefer the one supported by the detector; if the detector's
% class IS refuted, revise to the consistent class and report why.
%
% ASYMMETRY (the core of the method):
%   * appearance evidence  -> DEFEASIBLE support (may be wrong, esp. at range)
%   * kinematic violation  -> HARD refutation via integrity constraints
% This is what lets confident-but-wrong detector labels be corrected, with the
% violated constraint serving as the justification.
%
% Thresholds below are AUTHORED DOMAIN KNOWLEDGE (flight physics of small
% multirotors and of medium-sized birds), not values fitted to the data.
% No training is performed and no labels are consumed.
% ---------------------------------------------------------------------------

% --- observation facts (asserted per track window) --------------------------
:- dynamic f_det/2.        % f_det(Class, P)   appearance evidence + confidence
:- dynamic f_speed/1.      % f_speed(V)        mean speed over window (m/s)
:- dynamic f_speed_cv/1.   % f_speed_cv(CV)    coeff. of variation of speed
:- dynamic f_hover/1.      % f_hover(Frac)     fraction of window near-stationary
:- dynamic f_flap/1.       % f_flap(Bool)      wing-beat band oscillation present
:- dynamic f_turn_irreg/1. % f_turn_irreg(X)   std of heading-rate (rad/s)
:- dynamic f_straight/1.   % f_straight(Frac)  fraction of window near-zero turn
:- dynamic f_turn_bias/1.  % f_turn_bias(B)    MEAN SIGNED heading-rate (rad/s)

% --- domain thresholds (background theory T) --------------------------------
% Declared dynamic so that a SENSITIVITY ANALYSIS can override one value at a
% time (retract/assert) without editing the theory. The values below remain the
% authored defaults; the analysis exists to show the method's conclusions do
% not hinge on their exact setting.
:- dynamic thr/2.

thr(hover_speed,    1.0).   % m/s  below this the object is effectively holding
thr(hover_frac,     0.15).  % sustained hover if held for >15% of the window
thr(drone_v_max,  125.0).   % m/s  envelope ceiling incl. high-speed multirotors
thr(bird_v_max,    25.0).   % m/s  medium-bird level-flight ceiling
thr(turn_irreg,     0.35).  % rad/s heading-rate std above this = irregular
thr(straight_frac,  0.70).  % held straight for >70% of the window
thr(speed_cv_bird,  0.25).  % birds show larger speed fluctuation than rotors
thr(circling_bias,  0.10).  % rad/s persistent one-way turn = circling (thermalling)
thr(det_trust,      0.50).  % minimum detector confidence to count as support

% --- class signatures: what a class ENTAILS in the observation --------------
% Each clause is a human-auditable explanation of the kinematic evidence.

% A signature must be a POSITIVE ENTAILMENT: the class must imply that
% something specific is SEEN. An "absence" signature -- no wing beat and no
% irregularity -- is not an explanation but a default, and both classes entail
% it, so it cannot discriminate. Because `class(drone)` is tried first, such a
% clause silently claimed every quiet track for `drone`: at sigma = 0 it took
% 60 of 90 soaring/gliding birds, and the error VANISHED as noise grew (noise
% lifted turn_irreg past the threshold), i.e. the decision was driven by the
% noise level rather than by behaviour. Removed for that reason:
%
%   signature(drone, smooth_powered_flight) :-
%       f_flap(false), f_turn_irreg(T), thr(turn_irreg, TI), T =< TI.
%
% A quiet track now yields no drone signature, so the inference falls through
% to `accepted(not_refuted)` -- deferral to the detector, which is the correct
% behaviour when the kinematics carry no positive evidence.

signature(drone, holds_position) :-
    f_hover(H), thr(hover_frac, HF), H > HF.
signature(drone, sustained_straight_run) :-
    f_flap(false),
    f_straight(S), thr(straight_frac, SF), S > SF.

signature(bird, wing_beat_oscillation) :-
    f_flap(true).
signature(bird, irregular_manoeuvring) :-
    f_turn_irreg(T), thr(turn_irreg, TI), T > TI,
    f_hover(H), thr(hover_frac, HF), H =< HF.
signature(bird, fluctuating_airspeed) :-
    f_speed_cv(CV), thr(speed_cv_bird, CVB), CV > CVB,
    f_hover(H), thr(hover_frac, HF), H =< HF.
% Soaring / thermalling: a persistent ONE-WAY turn (circling), held REGULARLY
% and without wing beats. `f_turn_bias` (mean signed turn rate) separates this
% from a drone's straight run (bias ~ 0, low irregularity) and from flapping
% manoeuvring (bias ~ 0 but HIGH irregularity) -- a distinction `f_turn_irreg`
% alone cannot make.
% NOTE: rotorcraft also loiter in orbits, so circling SUPPORTS `bird` but is
% deliberately NOT a refutation of `drone`: ambiguous evidence must never
% trigger a revision.
signature(bird, soaring_circling) :-
    f_flap(false),
    f_turn_bias(B), thr(circling_bias, CB), abs(B) > CB,
    f_turn_irreg(T), thr(turn_irreg, TI), T =< TI.

% --- integrity constraints IC: hard refutations -----------------------------
% A class hypothesis is refuted when the kinematics are physically
% incompatible with it. These prune hypotheses regardless of the detector.

refutes(drone, flapping_flight) :-
    f_flap(true).
refutes(drone, beyond_rotor_envelope) :-
    f_speed(V), thr(drone_v_max, VM), V > VM.

% A robust constraint must not be gated by a fragile conjunct. `f_hover` is
% measured by NET DISPLACEMENT over a window, whose noise falls as sigma/sqrt(N)
% and which stays exact up to sigma = 1 m. Additionally demanding a small
% INSTANTANEOUS speed re-introduced the 1/dt noise amplification and silenced
% the constraint from sigma >= 0.3 m onwards, while adding no information:
% holding position over a window already entails not translating.
refutes(bird, sustained_hover) :-
    f_hover(H), thr(hover_frac, HF), H > HF.
refutes(bird, beyond_bird_envelope) :-
    f_speed(V), thr(bird_v_max, VM), V > VM.
% NOTE: a "long straight run without flapping" is deliberately NOT a refutation
% of `bird`: soaring birds in thermal or orographic lift hold a straight, beat-free
% glide for extended periods. Straight flight only SUPPORTS `drone`; it never
% refutes `bird`. Removing this constraint avoids systematic false revisions
% bird -> drone on soaring flight.

% --- candidate hypotheses ---------------------------------------------------
% A class is admissible iff no integrity constraint refutes it AND at least
% one of its signatures explains the observation.

class(drone). class(bird).

refuted(C, Why) :- refutes(C, Why), !.

admissible(C, Why) :-
    class(C),
    \+ refuted(C, _),
    signature(C, Why), !.

% --- abductive classification ----------------------------------------------
% classify(-Class, -Justification)
%
% 1. AGREEMENT   detector class is admissible          -> confirm
% 2. REVISION    detector class refuted, other admissible -> revise + reason
% 3. KINEMATIC   no usable detector evidence           -> decide on kinematics
% 4. UNKNOWN     nothing admissible / contradictory    -> abstain

classify(C, confirmed(C, Why)) :-
    f_det(C, P), thr(det_trust, PT), P >= PT,
    admissible(C, Why), !.

classify(C, revised(From, refuted_by(R), supported_by(Why))) :-
    f_det(From, P), thr(det_trust, PT), P >= PT,
    refuted(From, R),
    class(C), C \== From,
    admissible(C, Why), !.

% CONSERVATIVE DEFERRAL. If the detector's class is not refuted but no
% signature positively explains the observation -- the usual case when a fact
% needed by the signatures is not groundable and therefore not asserted -- the
% detector's label STANDS. The symbolic layer intervenes only when it has
% something to say; silence is not a reason to doubt perception.
classify(C, accepted(C, not_refuted)) :-
    f_det(C, P), thr(det_trust, PT), P >= PT,
    \+ refuted(C, _), !.

classify(C, kinematic_only(Why)) :-
    \+ ( f_det(_, P0), thr(det_trust, PT0), P0 >= PT0 ),
    admissible(C, Why), !.

classify(unknown, no_admissible_hypothesis) :-
    \+ admissible(_, _), !.

classify(unknown, contradictory_evidence).

% Convenience: class only.
classify(C) :- classify(C, _).
