% a4_track_alp.pl
% ---------------------------------------------------------------------------
% Abductive association of the attention window (Article 4, E10).
% Same structure as air_objects.pl of Article 3, but the hypotheses are about
% WHERE the held target is in the current frame, not about its class.
%
%   Delta* = argmin cost(Delta)  s.t.  T u Delta |= O,  T u Delta u IC not|- false
%
% Hypotheses for frame t (exactly one is abduced):
%   same(B)  -- candidate silhouette B is the held target;
%   unseen   -- the target is not visible in this frame (it coasts on the prediction).
% Integrity constraint (hard):
%   a target cannot leave its predicted position by more than TWO body lengths
%   in one frame. Body length = median size of the last accepted silhouettes,
%   floor 3 px (pixel quantisation). The coefficient 2 follows the measured
%   residual of the true target to the prediction (article 4b, revision): one
%   body length is exceeded in 7.6 % of the frames of the bird reference tracks
%   (break-up of the thresholded silhouette into body and wings), two body
%   lengths in 2.1 %; for UAVs in at most 0.5 %. The first version used one body
%   length (thr(body_mult, 1.0)).
% Preference (minimality): among the unrefuted same(B), the one closest to the
% prediction; unseen is abduced only when every same(B) is refuted, so a jump
% to a neighbour is never chosen over "not visible".
% Before the body length is known (fewer than 5 accepted silhouettes) the
% constraint is not grounded and the window gate of the base method applies.
% ---------------------------------------------------------------------------

:- dynamic f_pred/2.        % f_pred(X, Y)            predicted position
:- dynamic f_cand/4.        % f_cand(Id, X, Y, Size)  candidate silhouette in the window gate
:- dynamic f_body/1.        % f_body(L)               body length in px (asserted only when grounded)

thr(body_floor, 3.0).
thr(body_mult, 2.0).
thr(window_gate, 30.0).

residual(Id, R) :-
    f_cand(Id, X, Y, _), f_pred(PX, PY),
    R is sqrt((X - PX) ** 2 + (Y - PY) ** 2).

limit(L) :- f_body(B), !, thr(body_floor, F), thr(body_mult, M), L is M * max(B, F).
limit(G) :- thr(window_gate, G).            % constraint not groundable yet

refutes(same(Id), jump(R, L)) :-
    residual(Id, R), limit(L), R > L.

admissible(same(Id), R) :-
    f_cand(Id, _, _, _), \+ refutes(same(Id), _), residual(Id, R).

% explain(-Hypothesis, -Justification)
explain(same(Id), nearest_unrefuted(R)) :-
    findall(R0-I0, admissible(same(I0), R0), L), L \== [], !,
    keysort(L, [R-Id | _]).
explain(unseen, all_refuted(Why)) :-
    findall(I-W, refutes(same(I), W), Why), Why \== [], !.
explain(unseen, no_candidate).
