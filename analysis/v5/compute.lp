#script (python) 

import datetime
import string
import sys 
import clingo 
import os

from clingo.symbol import Function

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
    
    # debug = False 
    
    debug = True
    listAll = []
    counter = 0 
    maxLength = 0 

    # compile the program 
        
    prg.ground([("base",[])]) 


    while (True):  
   
        ret = prg.solve(on_model=process_model) 
    
        if (not ret.satisfiable) : break 
   
        print("Answer set counter # ", counter, curr_as)  
    
        keep_atoms = [str(x) for x in curr_as if x.match("ok", 2)] 
         
        print("Keep ", keep_atoms)  
 
        eliminate_atoms = ["ok("+str(x.arguments[0])+","+str(x.arguments[1])+")" for x in curr_as if x.match("nok", 2)] 
        
        print("Elimination ", eliminate_atoms) 
    
        if (debug) :  print ("List of all OK elements *********** : {}".format(keep_atoms))  

        listAll.append(keep_atoms) 
         
        if (debug): print ("Current list of all answer sets", len(listAll))  

        # constraint = ":- " + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1" +  ","+ ''.join(["not ok("+str(x)+")," for x in eliminate_atoms]) + "1==1." 

        # constraint = ":- 1 {" + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1}" + str(len(keep_atoms)+1)+ ","+ ''.join(["not ok("+str(x)+")," for x in eliminate_atoms]) + "1==1." 

        if len(eliminate_atoms) > 0 :
              constraint = ":- " + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1" +  ","+ ''.join(["not "+str(x)+"," for x in eliminate_atoms]) + "1==1."
              # constraint = ":- 1 {" + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1}" + str(len(keep_atoms)+1)+ ","+ ''.join(["not "+str(x)+"," for x in eliminate_atoms]) + "1==1." 
              # constraint = ":- "+ str(len(keep_atoms)+1)+"{" + ''.join([str(x)+";" for x in keep_atoms]) + " 1==1}" + str(len(keep_atoms)+1)+ ","+ ''.join(["not "+str(x)+"," for x in eliminate_atoms]) + "1==1." 
              # print("here\n", len(eliminate_atoms))
        else : 
              break 
    
        if (debug):  print("Constraint ... \n ",  constraint) 

        prg.add("constraints", [],  constraint)
    
        prg.ground([("constraints",[])])
     
        counter = counter + 1 

        # prg.release_external(Function("first_check"))
             
    print("\n\n All optimal answer sets:",  listAll, len(listAll)) 
    
    lengthAS = [len(x) for x in listAll] 
    
    print(lengthAS) 
     
#end.
 

