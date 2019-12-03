import pandas as pd
import json
import numpy as np


class myProblem:
    with open(r'C:\Users\Tom\Desktop\example.json') as f:
        data = json.load(f)
    name = data["name"]
    kommentar = data["kommentar"]
    reihenfolge = data["reihenfolge"]
    bearbeitungszeiten = data["bearbeitungszeiten"]
    zielfunktion = data["zielfunktion"]
    if(zielfunktion == "Lmax"):
        liefertermin = data["liefertermin"]
    n = len(reihenfolge[0])
    m = len(reihenfolge)


p1 = myProblem()



#-----------------------

#für den aktuellen State
time = 0
maxTime = np.matrix.sum(np.matrix(p1.bearbeitungszeiten))
belegung = np.zeros((p1.m, p1.n))       #null = Maschine frei, nicht null = Zeit bis Maschine frei
zuBearbeiten = p1.bearbeitungszeiten    #null = ist/wird bearbeitet, nicht null muss bearbeitet werden
nextMaschine = p1.reihenfolge[0]        #nextMaschine[i] gibt nächste Machine für Aufrtrag i. Falls -1 ist, ist Aufrag i fertig
#q = np.zeros(, maxTime))
bestValue = -1
bestConfig = np.zeros((p1.m, maxTime))   #Gant Diagramm
currentConfig = np.zeros((p1.m, maxTime))

print()

def learn():
    while(not np.all(nextMaschine[i] == -1 for i in range(0,p1.n))):
        options = getCurrentOptions()
        while(len(options) > 0):
            rand = np.random.randint(len(options)) #bisher nur random
            schedule(options[rand])
            options = getCurrentOptions()
        ..



def getCurrentOptions():
    result = []
    for i in range(0, p1.n):
        j = nextMaschine[i]
        if(j != -1):
            if(belegung[j][i] == 0):
                result.append((i+1,j))
    return result

def schedule(t):
    i = t[0]
    j = t[1]
    #update all





#for i in range(100):
#    learn()

#print(getCurrentOptions())

#print(bestConfig)


