#script (python) 

import datetime
import string
import sys 
import clingo 
import os

from clingo.symbol import Function

global curr_as 
global listAll 
global debug 


def main(prg):

    global listAll   
    global debug 
    
    # debug = False     
    debug = True
    
    listAll = []

    # set option  
        
    prg.configuration.solve.opt_mode = "optN"
            
    # compile the program 
    
    prg.ground([("base",[])]) 

    with prg.solve(yield_=True) as handle: 
         for m in handle: 
                if m.optimality_proven == True: 
                     listAll.append([str(x) for x in m.symbols(atoms=True) if x.match("ok", 2)]) 

    print("\n\n All optimal answer sets:",  listAll, len(listAll)) 

    return  
     
#end.
 

