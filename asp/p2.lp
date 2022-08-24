% Type requirements 
2{bad(locatedIn(P,Q)); bad(org(P))}:- atom(locatedIn(P,Q)), atom(org(P)).
2{bad(locatedIn(P,Q)); bad(peop(P))} :- atom(locatedIn(P,Q)), atom(peop(P)).
2{bad(locatedIn(P,Q)); bad(other(P))} :- atom(locatedIn(P,Q)), atom(other(P)).
2{bad(locatedIn(P,Q)); bad(org(Q))}:- atom(locatedIn(P,Q)), atom(org(Q)).
2{bad(locatedIn(P,Q)); bad(peop(Q))} :- atom(locatedIn(P,Q)), atom(peop(Q)).
2{bad(locatedIn(P,Q)); bad(other(Q))}:- atom(locatedIn(P,Q)), atom(other(Q)).

2{bad(orgbasedIn(P,Q));bad(loc(P))}:- atom(orgbasedIn(P,Q)), atom(loc(P)).
2{bad(orgbasedIn(P,Q)); bad(peop(P))} :- atom(orgbasedIn(P,Q)), atom(peop(P)).
2{bad(orgbasedIn(P,Q)); bad(other(P))}:- atom(orgbasedIn(P,Q)), atom(other(P)).
2{bad(orgbasedIn(P,Q)); bad(org(Q))}:- atom(orgbasedIn(P,Q)), atom(org(Q)).
2{bad(orgbasedIn(P,Q)); bad(peop(Q))} :- atom(orgbasedIn(P,Q)), atom(peop(Q)).
2{bad(orgbasedIn(P,Q)); bad(other(Q))}:- atom(orgbasedIn(P,Q)), atom(other(Q)).

2{bad(liveIn(P,Q));bad(loc(P))}:- atom(liveIn(P,Q)), atom(loc(P)).
2{bad(liveIn(P,Q)); bad(org(P))}:- atom(liveIn(P,Q)), atom(org(P)).
2{bad(liveIn(P,Q)); bad(other(P))}:- atom(liveIn(P,Q)), atom(other(P)).
2{bad(liveIn(P,Q)); bad(peop(Q))} :- atom(liveIn(P,Q)), atom(peop(Q)).
2{bad(liveIn(P,Q)); bad(org(Q))}:- atom(liveIn(P,Q)), atom(org(Q)).
2{bad(liveIn(P,Q)); bad(other(Q))}:- atom(liveIn(P,Q)), atom(other(Q)).

2{bad(workFor(P,Q));bad(loc(P))}:- atom(workFor(P,Q)), atom(loc(P)).
2{bad(workFor(P,Q)); bad(org(P))}:- atom(workFor(P,Q)), atom(org(P)).
2{bad(workFor(P,Q)); bad(other(P))}:- atom(workFor(P,Q)), atom(other(P)).
2{bad(workFor(P,Q)); bad(peop(Q))} :- atom(workFor(P,Q)), atom(peop(Q)).
2{bad(workFor(P,Q));bad(loc(Q))}:- atom(workFor(P,Q)), atom(loc(Q)).
2{bad(workFor(P,Q)); bad(other(Q))}:- atom(workFor(P,Q)), atom(other(Q)).

2{bad(kill(P,Q));bad(loc(P))}:- atom(kill(P,Q)), atom(loc(P)).
2{bad(kill(P,Q)); bad(org(P))}:- atom(kill(P,Q)), atom(org(P)).
2{bad(kill(P,Q)); bad(other(P))}:- atom(kill(P,Q)), atom(other(P)).
2{bad(kill(P,Q));bad(loc(Q))}:- atom(kill(P,Q)), atom(loc(Q)).
2{bad(kill(P,Q));bad(org(Q))}:- atom(kill(P,Q)), atom(org(Q)).
2{bad(kill(P,Q)); bad(other(Q))}:- atom(kill(P,Q)), atom(other(Q)).


% Other requirements
bad(liveIn(X,X)):- atom(liveIn(X,X)).
bad(locatedIn(X,X)):- atom(locatedIn(X,X)).
bad(orgbasedIn(X,X)):- atom(orgbasedIn(X,X)).
bad(workFor(X,X)):- atom(workFor(X,X)).
bad(kill(X,X)):- atom(kill(X,X)).

2{bad(peop(X)); bad(loc(X))} :- atom(peop(X)), atom(loc(X)).
2{bad(peop(X)); bad(org(X))} :- atom(peop(X)), atom(org(X)).
2{bad(peop(X)); bad(other(X))}:- atom(peop(X)), atom(other(X)).
2{bad(loc(X)); bad(org(X))}:- atom(loc(X)), atom(org(X)).
2{bad(loc(X)); bad(other(X))}:- atom(loc(X)), atom(other(X)).
2{bad(org(X)); bad(other(X))}:- atom(org(X)), atom(other(X)).

% Type requirements 
:- ok(locatedIn(P,Q)), ok(org(P)).
:- ok(locatedIn(P,Q)), ok(peop(P)).
:- ok(locatedIn(P,Q)), ok(other(P)).
:- ok(locatedIn(P,Q)), ok(org(Q)).
:- ok(locatedIn(P,Q)), ok(peop(Q)).
:- ok(locatedIn(P,Q)), ok(other(Q)).

:- ok(orgbasedIn(P,Q)), ok(loc(P)).
:- ok(orgbasedIn(P,Q)), ok(peop(P)).
:- ok(orgbasedIn(P,Q)), ok(other(P)).
:- ok(orgbasedIn(P,Q)), ok(org(Q)).
:- ok(orgbasedIn(P,Q)), ok(peop(Q)).
:- ok(orgbasedIn(P,Q)), ok(other(Q)).

:- ok(liveIn(P,Q)), ok(loc(P)).
:- ok(liveIn(P,Q)), ok(org(P)).
:- ok(liveIn(P,Q)), ok(other(P)).
:- ok(liveIn(P,Q)), ok(peop(Q)).
:- ok(liveIn(P,Q)), ok(org(Q)).
:- ok(liveIn(P,Q)), ok(other(Q)).

:- ok(workFor(P,Q)), ok(loc(P)).
:- ok(workFor(P,Q)), ok(org(P)).
:- ok(workFor(P,Q)), ok(other(P)).
:- ok(workFor(P,Q)), ok(peop(Q)).
:- ok(workFor(P,Q)), ok(loc(Q)).
:- ok(workFor(P,Q)), ok(other(Q)).

:- ok(kill(P,Q)), ok(loc(P)).
:- ok(kill(P,Q)), ok(org(P)).
:- ok(kill(P,Q)), ok(other(P)).
:- ok(kill(P,Q)), ok(loc(Q)).
:- ok(kill(P,Q)), ok(org(Q)).
:- ok(kill(P,Q)), ok(other(Q)).


% Other requirements
:- ok(liveIn(X,X)).
:- ok(locatedIn(X,X)).
:- ok(orgbasedIn(X,X)).
:- ok(workFor(X,X)).
:- ok(kill(X,X)).

:- ok(peop(X)), ok(loc(X)).
:- ok(peop(X)), ok(org(X)).
:- ok(peop(X)), ok(other(X)).
:- ok(loc(X)), ok(org(X)).
:- ok(loc(X)), ok(other(X)).
:- ok(org(X)), ok(other(X)).

ok(X)  :- atom(X), not bad(X).

{ ok(X) } :- atom(X), bad(X).

nok(X) :- atom(X), not ok(X). 

nOfOKAtoms(N) :- N = #count{1,X : ok(X),atom(X)}.

#maximize {N : nOfOKAtoms(N)}.

#show ok/1.
#show nOfOKAtoms/1.
#show bad/1.  

