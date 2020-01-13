import json
import numpy as np
import os

fixedSize = 100
name = str(fixedSize)+'x'+str(fixedSize)+'-problem'
comment ='a '+str(fixedSize)+'x'+str(fixedSize)+'-type problem'
problem_size = (fixedSize,fixedSize)

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname,name+".json")




def generate() :
    temp_order = np.random.randint(low=1, high=20,size=problem_size, dtype=int)
    tmp_ptime = np.zeros(problem_size, dtype=int)
    
    for i in range(0,problem_size[1]) :
        tmp_permute = np.random.permutation(problem_size[0])
        for j in range(0,problem_size[0]) :
            tmp_ptime[j][i] = tmp_permute[j]

    return tmp_ptime,temp_order

problem_order, problem_ptime = generate()
problem_order_list = problem_order.tolist()
problem_ptime_list = problem_ptime.tolist()


data = {
    'name' : name,
    'kommentar' : comment,
    'reihenfolge' : problem_order_list,
    'bearbeitungszeiten' : problem_ptime_list,
    'zielfunktion' : 'Cmax'
}


if(os.path.isfile(filename)):
    os.remove(filename)

with open(filename, 'x') as f:
    json.dump(data,f,indent=1)
    f.write("\n")


