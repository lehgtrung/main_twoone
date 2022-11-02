:- locatedIn(P1,P2), orgbasedIn(O,P1), not orgbasedIn(O, P2).
:- locatedIn(P1,P2), locatedIn(P2,P3), not locatedIn(P1, P3).
:- liveIn(X,P1), locatedIn(P1,P2), not liveIn(X,P2).
:- workFor(X,O), orgbasedIn(O,P), not liveIn(X,P).
:- liveIn(X,P), workFor(X,O), not orgbasedIn(O,P).


% Triplets
% orgbasedIn(O, P2) :- locatedIn(P1,P2), orgbasedIn(O,P1).
% locatedIn(P1, P3) :- locatedIn(P1,P2), locatedIn(P2,P3).
% liveIn(X,P2) :- liveIn(X,P1), locatedIn(P1,P2).

% Not absolutely correct, we assume no one works remotely
% liveIn(X,P) :- workFor(X,O), orgbasedIn(O,P).
% orgbasedIn(O,P) :- liveIn(X,P), workFor(X,O).


