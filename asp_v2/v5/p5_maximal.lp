% Type requirements -- if the body is satisfied then one of the atom in the body is incorrect

2{bad(locatedIn(P,Q),I); bad(org(P),I)}:- atom(locatedIn(P,Q),I), atom(org(P),I).
2{bad(locatedIn(P,Q),I); bad(peop(P),I)} :- atom(locatedIn(P,Q),I), atom(peop(P),I).
2{bad(locatedIn(P,Q),I); bad(other(P),I)} :- atom(locatedIn(P,Q),I), atom(other(P),I).
2{bad(locatedIn(P,Q),I); bad(org(Q),I)}:- atom(locatedIn(P,Q),I), atom(org(Q),I).
2{bad(locatedIn(P,Q),I); bad(peop(Q),I)} :- atom(locatedIn(P,Q),I), atom(peop(Q),I).
2{bad(locatedIn(P,Q),I); bad(other(Q),I)}:- atom(locatedIn(P,Q),I), atom(other(Q),I).

2{bad(orgbasedIn(P,Q),I);bad(loc(P),I)}:- atom(orgbasedIn(P,Q),I), atom(loc(P),I).
2{bad(orgbasedIn(P,Q),I); bad(peop(P),I)} :- atom(orgbasedIn(P,Q),I), atom(peop(P),I).
2{bad(orgbasedIn(P,Q),I); bad(other(P),I)}:- atom(orgbasedIn(P,Q),I), atom(other(P),I).
2{bad(orgbasedIn(P,Q),I); bad(org(Q),I)}:- atom(orgbasedIn(P,Q),I), atom(org(Q),I).
2{bad(orgbasedIn(P,Q),I); bad(peop(Q),I)} :- atom(orgbasedIn(P,Q),I), atom(peop(Q),I).
2{bad(orgbasedIn(P,Q),I); bad(other(Q),I)}:- atom(orgbasedIn(P,Q),I), atom(other(Q),I).

2{bad(liveIn(P,Q),I);bad(loc(P),I)}:- atom(liveIn(P,Q),I), atom(loc(P),I).
2{bad(liveIn(P,Q),I); bad(org(P),I)}:- atom(liveIn(P,Q),I), atom(org(P),I).
2{bad(liveIn(P,Q),I); bad(other(P),I)}:- atom(liveIn(P,Q),I), atom(other(P),I).
2{bad(liveIn(P,Q),I); bad(peop(Q),I)} :- atom(liveIn(P,Q),I), atom(peop(Q),I).
2{bad(liveIn(P,Q),I); bad(org(Q),I)}:- atom(liveIn(P,Q),I), atom(org(Q),I).
2{bad(liveIn(P,Q),I); bad(other(Q),I)}:- atom(liveIn(P,Q),I), atom(other(Q),I).

2{bad(workFor(P,Q),I);bad(loc(P),I)}:- atom(workFor(P,Q),I), atom(loc(P),I).
2{bad(workFor(P,Q),I); bad(org(P),I)}:- atom(workFor(P,Q),I), atom(org(P),I).
2{bad(workFor(P,Q),I); bad(other(P),I)}:- atom(workFor(P,Q),I), atom(other(P),I).
2{bad(workFor(P,Q),I); bad(peop(Q),I)} :- atom(workFor(P,Q),I), atom(peop(Q),I).
2{bad(workFor(P,Q),I);bad(loc(Q),I)}:- atom(workFor(P,Q),I), atom(loc(Q),I).
2{bad(workFor(P,Q),I); bad(other(Q),I)}:- atom(workFor(P,Q),I), atom(other(Q),I).

2{bad(kill(P,Q),I);bad(loc(P),I)}:- atom(kill(P,Q),I), atom(loc(P),I).
2{bad(kill(P,Q),I); bad(org(P),I)}:- atom(kill(P,Q),I), atom(org(P),I).
2{bad(kill(P,Q),I); bad(other(P),I)}:- atom(kill(P,Q),I), atom(other(P),I).
2{bad(kill(P,Q),I);bad(loc(Q),I)}:- atom(kill(P,Q),I), atom(loc(Q),I).
2{bad(kill(P,Q),I);bad(org(Q),I)}:- atom(kill(P,Q),I), atom(org(Q),I).
2{bad(kill(P,Q),I); bad(other(Q),I)}:- atom(kill(P,Q),I), atom(other(Q),I).


% Other requirements
bad(liveIn(X,X),I):- atom(liveIn(X,X),I).
bad(locatedIn(X,X),I):- atom(locatedIn(X,X),I).
bad(orgbasedIn(X,X),I):- atom(orgbasedIn(X,X),I).
bad(workFor(X,X),I):- atom(workFor(X,X),I).
bad(kill(X,X),I):- atom(kill(X,X),I).

2{bad(peop(X),I); bad(loc(X),I)} :- atom(peop(X),I), atom(loc(X),I).
2{bad(peop(X),I); bad(org(X),I)} :- atom(peop(X),I), atom(org(X),I).
2{bad(peop(X),I); bad(other(X),I)}:- atom(peop(X),I), atom(other(X),I).
2{bad(loc(X),I); bad(org(X),I)}:- atom(loc(X),I), atom(org(X),I).
2{bad(loc(X),I); bad(other(X),I)}:- atom(loc(X),I), atom(other(X),I).
2{bad(org(X),I); bad(other(X),I)}:- atom(org(X),I), atom(other(X),I).

% Type requirements --
% the body of a constraint should not be satisfied

:- ok(locatedIn(P,Q),I), ok(org(P),I).
:- ok(locatedIn(P,Q),I), ok(peop(P),I).
:- ok(locatedIn(P,Q),I), ok(other(P),I).
:- ok(locatedIn(P,Q),I), ok(org(Q),I).
:- ok(locatedIn(P,Q),I), ok(peop(Q),I).
:- ok(locatedIn(P,Q),I), ok(other(Q),I).

:- ok(orgbasedIn(P,Q),I), ok(loc(P),I).
:- ok(orgbasedIn(P,Q),I), ok(peop(P),I).
:- ok(orgbasedIn(P,Q),I), ok(other(P),I).
:- ok(orgbasedIn(P,Q),I), ok(org(Q),I).
:- ok(orgbasedIn(P,Q),I), ok(peop(Q),I).
:- ok(orgbasedIn(P,Q),I), ok(other(Q),I).

:- ok(liveIn(P,Q),I), ok(loc(P),I).
:- ok(liveIn(P,Q),I), ok(org(P),I).
:- ok(liveIn(P,Q),I), ok(other(P),I).
:- ok(liveIn(P,Q),I), ok(peop(Q),I).
:- ok(liveIn(P,Q),I), ok(org(Q),I).
:- ok(liveIn(P,Q),I), ok(other(Q),I).

:- ok(workFor(P,Q),I), ok(loc(P),I).
:- ok(workFor(P,Q),I), ok(org(P),I).
:- ok(workFor(P,Q),I), ok(other(P),I).
:- ok(workFor(P,Q),I), ok(peop(Q),I).
:- ok(workFor(P,Q),I), ok(loc(Q),I).
:- ok(workFor(P,Q),I), ok(other(Q),I).

:- ok(kill(P,Q),I), ok(loc(P),I).
:- ok(kill(P,Q),I), ok(org(P),I).
:- ok(kill(P,Q),I), ok(other(P),I).
:- ok(kill(P,Q),I), ok(loc(Q),I).
:- ok(kill(P,Q),I), ok(org(Q),I).
:- ok(kill(P,Q),I), ok(other(Q),I).

%% Other requirements

:- ok(liveIn(X,X),I).
:- ok(locatedIn(X,X),I).
:- ok(orgbasedIn(X,X),I).
:- ok(workFor(X,X),I).
:- ok(kill(X,X),I).

:- ok(peop(X),I), ok(loc(X),I).
:- ok(peop(X),I), ok(org(X),I).
:- ok(peop(X),I), ok(other(X),I).
:- ok(loc(X),I), ok(org(X),I).
:- ok(loc(X),I), ok(other(X),I).
:- ok(org(X),I), ok(other(X),I).


%% Determine whether set of atom I (atom(X, I)) is inconsistent

%% inconsistency(I) :- atom(X, I), bad(X, I).

%% Uncommented the line below if we do not want to use the output
%% whenever model 2 is inconsistent.

%% :- inconsistency(2).

%%% an atom should be kept if it does not violate any type constraint
%%% e.g., peop(x) is okay if there is no other atom indicating that it is a location, an organization, or other entity
%%%                      and x is not in any relation such as workfor(P,x) ...

ok(X, 1)  :- atom(X, 1), not bad(X, 1).

%%% an atom could be kept if it contributes to the violation of some constraint
%%% or it is an inferred atom

{ ok(X, 1) } :- atom(X, 1), bad(X, 1).

{ ok(X, 1) } :- inf_atom(X, 1), bad(X, 1).

%%% when an atom is not kept then it is not okay

nok(X, 1) :- atom(X, 1), not ok(X, 1).
nok(X, 1) :- inf_atom(X, 1), not ok(X, 1).

%%% computing set difference

%%inSetDifference(X) :- atom(X, 1), ok(X, 1), not atom(X, 2).
%%inSetDifference(X) :- atom(X, 2), nok(X, 1).

inSetDifference(X) :- ok(X, 1), not atom(X, 1).
inSetDifference(X) :- atom(X, 1), nok(X, 1).

%%% keeping the maximal number of atoms that can be kept

%% nOfAtoms(N) :- N = #count{1,X : inSetDifference(X)}.

%% #minimize {N : nOfAtoms(N)}.

nOfAtoms(N) :- N = #count{1,X : ok(X)}.
#maximize {N : nOfAtoms(N)}.


#show ok/2.
%% #show nOfAtoms/1.
%% #show bad/1.
%% #show atom/1.
%% #show dependency/2.
%% #show rule/3.

%%%%%%%%%%% inference rules %%%%%%%%%%%%%%%%%%%%

rule(locatedIn(P1,P2), orgbasedIn(O,P1), orgbasedIn(O, P2)):-
	atom(locatedIn(P1,P2),1), atom(orgbasedIn(O,P1),1), P1 != P2.

rule(locatedIn(P1,P2), locatedIn(P2,P3), locatedIn(P1, P3)):-
	atom(locatedIn(P1,P2),1), atom(locatedIn(P2,P3),1), P1 != P2, P2 != P3.

rule(liveIn(X,P1), locatedIn(P1,P2), liveIn(X,P2)):-
	atom(liveIn(X,P1),1), atom(locatedIn(P1,P2),1), P1!=P2.

rule(workFor(X,O), orgbasedIn(O,P),liveIn(X,P)):-
	atom(workFor(X,O),1), atom(orgbasedIn(O,P),1).

rule(liveIn(X,P), workFor(X,O), orgbasedIn(O,P)):-
	atom(liveIn(X,P),1), atom(workFor(X,O),1).



%%%%% ASP for working with inference rules %%%%%%%%%%%%%%%%

6 {bad(X,1); bad(Y,1); bad(Z,1); inf_atom(Z,1);
    dependency(X, Z); dependency(Y, Z) } :-
       	rule(X, Y, Z), atom(X, 1), atom(Y, 1), not atom(Z,1).

:- ok(Y, 1), inf_atom(Y,1), dependency(X,Y), not ok(X, 1).
:- rule(X, Y, Z), ok(X, 1), ok(Y, 1), not ok(Z, 1).


%% Trung: Restrict entity to be head/tail of 2 different relations
%% locatedIn/orgbasedIn/liveIn/workFor/kill
%:- ok(locatedIn(X,_), 1), ok(orgbasedIn(X,_), 1).
%:- ok(locatedIn(X,_), 1), ok(liveIn(X,_), 1).
%:- ok(locatedIn(X,_), 1), ok(workFor(X,_), 1).
%:- ok(locatedIn(X,_), 1), ok(kill(X,_), 1).

%:- ok(orgbasedIn(X,_), 1), ok(liveIn(X,_), 1).
%:- ok(orgbasedIn(X,_), 1), ok(workFor(X,_), 1).
%:- ok(orgbasedIn(X,_), 1), ok(kill(X,_), 1).


%% Trung: For every x, if ok(x) then selected(x)
%selected(X, 1) :- ok(X, 1).



%% Trung: infer types for relations only after they are selected
%% locatedIn/orgbasedIn/liveIn/workFor/kill

ok(loc(P1), 1) :- ok(locatedIn(P1,P2),1).
ok(loc(P2), 1) :- ok(locatedIn(P1,P2),1).
ok(org(O), 1) :- ok(orgbasedIn(O,P),1).
ok(loc(P), 1) :- ok(orgbasedIn(O,P),1).
ok(peop(X), 1) :- ok(liveIn(X,P),1).
ok(loc(P), 1) :- ok(liveIn(X,P),1).
ok(peop(X), 1) :- ok(workFor(X,O),1).
ok(org(O), 1) :- ok(workFor(X,O),1).
ok(peop(X), 1) :- ok(kill(X,Y),1).
ok(peop(Y), 1) :- ok(kill(X,Y),1).
