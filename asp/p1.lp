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


% More regulations
%:- ok(locatedIn(X,_)), ok(orgbasedIn(X,_)).
%:- ok(locatedIn(X,_)), ok(liveIn(X,_)).
%:- ok(locatedIn(X,_)), ok(workFor(X,_)).
%:- ok(locatedIn(X,_)), ok(kill(X,_)).

%:- ok(orgbasedIn(X,_)), ok(liveIn(X,_)).
%:- ok(orgbasedIn(X,_)), ok(workFor(X,_)).
%:- ok(orgbasedIn(X,_)), ok(kill(X,_)).

%:- ok(locatedIn(_,X)), ok(workFor(_,X)).
%:- ok(locatedIn(_,X)), ok(kill(_,X)).

%:- ok(orgbasedIn(_,X)), ok(workFor(_,X)).
%:- ok(orgbasedIn(_,X)), ok(kill(_,X)).

%:- ok(liveIn(_,X)), ok(workFor(_,X)).
%:- ok(liveIn(_,X)), ok(kill(_,X)).

%:- ok(workFor(_,X)), ok(kill(_,X)).

% Directional regulations
%:- ok(locatedIn(X,Y)), ok(locatedIn(Y,X)).
%:- ok(orgbasedIn(X,Y)), ok(orgbasedIn(Y,X)).
%:- ok(liveIn(X,Y)), ok(liveIn(Y,X)).
%:- ok(workFor(X,Y)), ok(workFor(Y,X)).
%:- ok(kill(X,Y)), ok(kill(Y,X)).


% We don't want a relation if its head or tail entity is not selected
%:- atom(orgbasedIn(X, Y)), ok(orgbasedIn(X, Y)), not ok(org(X)).
%:- atom(orgbasedIn(X, Y)), ok(orgbasedIn(X, Y)), not ok(loc(Y)).

%:- atom(locatedIn(X, Y)), ok(locatedIn(X, Y)), not ok(loc(X)).
%:- atom(locatedIn(X, Y)), ok(locatedIn(X, Y)), not ok(loc(Y)).

%:- atom(liveIn(X, Y)), ok(liveIn(X, Y)), not ok(peop(X)).
%:- atom(liveIn(X, Y)), ok(liveIn(X, Y)), not ok(loc(Y)).

%:- atom(workFor(X, Y)), ok(workFor(X, Y)), not ok(peop(X)).
%:- atom(workFor(X, Y)), ok(workFor(X, Y)), not ok(org(Y)).

%:- atom(kill(X, Y)), ok(kill(X, Y)), not ok(peop(X)).
%:- atom(kill(X, Y)), ok(kill(X, Y)), not ok(peop(Y)).

% We don't want a standalone entity (how to filter standalone entities)
%:- atom(org(X)), ok(org(X)), #count{1,Y : ok(orgbasedIn(X, Y))}==0, #count{1,Y : ok(workFor(Y, X))}==0.

%:- atom(loc(X)), ok(loc(X)), #count{1,Y : ok(locatedIn(X, Y))}==0,
%#count{1,Y : ok(locatedIn(Y, X))}==0, #count{1,Y : ok(orgbasedIn(Y, X))}==0,
%#count{1,Y : ok(liveIn(Y, X))}==0.

%:- atom(peop(X)), ok(peop(X)), #count{1,Y : ok(liveIn(X, Y))}==0,
%#count{1,Y : ok(kill(X, Y))}==0,
%#count{1,Y : ok(kill(Y, X))}==0.


{ ok(X) } :- atom(X).

% 07/27/2022
nok(X) :- atom(X), not ok(X).

nOfOKAtoms(N) :- N = #count{1,X : ok(X),atom(X)}.

#maximize {N : nOfOKAtoms(N)}.

#show ok/1.
#show nOfOKAtoms/1.

