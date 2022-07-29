#script (python)

import datetime
import string
import sys
import clingo
import os

global curr_as
global maxLength
global listAll
global debug

def process_model(m):
    global curr_as
    global listAll

    curr_as = m.symbols(atoms=True)

    if (debug): print ("List of all elements: {}".format(curr_as), len(curr_as))


def main(prg):

    global listAll
    global debug
    # global curr_as

    debug = False

    # debug = True
    listAll = []
    counter = 0
    maxLength = 0

    # compile the program

    prg.ground([("base",[])])

    # get first answer set, process

    # print("Answer set  ************************* ")

    # prg.solve(on_model=process_model)

    # print("ret.satisfaiable " , ret.satisfiable, curr_as)

    # return

    while (True):

        ret = prg.solve(on_model=process_model)

        if (not ret.satisfiable) : break

        print("Answer set counter # ", counter, curr_as)

        keep_atoms = [str(x) for x in curr_as if x.match("ok", 1)]

        eliminate_atoms = [str(x.arguments[0]) for x in curr_as if x.match("nok", 1)]

        print("Elimination ", eliminate_atoms)

        if (debug) :  print ("List of all OK elements *********** : {}".format(keep_atoms))

        listAll.append(keep_atoms)


        # if counter == 0 or len(keep_atoms) >= maxLength :
        #        maxLength =  len(keep_atoms)
        #        listAll.append(keep_atoms)
        # else :   break

        if (debug): print ("Current list of all answer sets", len(listAll))

        if len(eliminate_atoms) > 0 :
              constraint = ":- 1{" + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1}" + str(len(keep_atoms)+1)+ ","+ ''.join(["not ok("+str(x)+")," for x in eliminate_atoms]) + "1==1."
        else :
              break

        if (debug):  print("Constraint ... \n ",  constraint)

        prg.add("constraints", [],  constraint)

        prg.ground([("constraints",[])])

        counter = counter + 1

    # print("\n\n All optimal answer sets:",  listAll, len(listAll))
    newListAll = []
    for answerset in listAll:
        _list = []
        for atom in answerset:
            atom = str(atom)
            _list.append(atom)
        newListAll.append(_list)
    print(newListAll)

#end.
