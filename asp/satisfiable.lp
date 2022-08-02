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


