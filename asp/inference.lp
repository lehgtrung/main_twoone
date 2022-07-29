% Type requirements
:- locatedIn(P,Q), org(P).
:- locatedIn(P,Q), peop(P).
:- locatedIn(P,Q), other(P).
:- locatedIn(P,Q), org(Q).
:- locatedIn(P,Q), peop(Q).
:- locatedIn(P,Q), other(Q).

:- orgbasedIn(P,Q), loc(P).
:- orgbasedIn(P,Q), peop(P).
:- orgbasedIn(P,Q), other(P).
:- orgbasedIn(P,Q), org(Q).
:- orgbasedIn(P,Q), peop(Q).
:- orgbasedIn(P,Q), other(Q).

:- liveIn(P,Q), loc(P).
:- liveIn(P,Q), org(P).
:- liveIn(P,Q), other(P).
:- liveIn(P,Q), peop(Q).
:- liveIn(P,Q), org(Q).
:- liveIn(P,Q), other(Q).

:- workFor(P,Q), loc(P).
:- workFor(P,Q), org(P).
:- workFor(P,Q), other(P).
:- workFor(P,Q), peop(Q).
:- workFor(P,Q), loc(Q).
:- workFor(P,Q), other(Q).

:- kill(P,Q), loc(P).
:- kill(P,Q), org(P).
:- kill(P,Q), other(P).
:- kill(P,Q), loc(Q).
:- kill(P,Q), org(Q).
:- kill(P,Q), other(Q).


% Other requirements
:- liveIn(X,X).
:- locatedIn(X,X).
:- orgbasedIn(X,X).
:- workFor(X,X).
:- kill(X,X).

:- peop(X), loc(X).
:- peop(X), org(X).
:- peop(X), other(X).
:- loc(X), org(X).
:- loc(X), other(X).
:- org(X), other(X).

% Type inference

loc(P) :- locatedIn(P,Q).
loc(Q) :- locatedIn(P,Q).

org(O) :- orgbasedIn(O,P).
loc(P) :- orgbasedIn(O,P).

peop(X) :- liveIn(X,P).
loc(P) :- liveIn(X,P).

peop(X) :- workFor(X,O).
org(O) :- workFor(X,O).

peop(X) :- kill(X,Y).
peop(Y) :- kill(X,Y).


% Triplets
orgbasedIn(O, P2) :- locatedIn(P1,P2), orgbasedIn(O,P1).
locatedIn(P1, P3) :- locatedIn(P1,P2), locatedIn(P2,P3).
liveIn(X,P2) :- liveIn(X,P1), locatedIn(P1,P2).

% Not absolutely correct, we assume no one works remotely
liveIn(X,P) :- workFor(X,O), orgbasedIn(O,P).
orgbasedIn(O,P) :- liveIn(X,P), workFor(X,O).

