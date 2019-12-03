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
maxRandQValue = 10                      #Q Table zufällig zwischen 0 und 10
orderPointer = np.zeros(p1.m)           #Zeigt auf die Stelle in der Q tabelle der Machinen des index, der als nächsts dran ist
q = np.array(p1.m)
for i in range(0, p1.m):
        q[i] = np.randint(high=maxRandQValue,size=(p1.n, p1.n))

bestValue = -1
bestConfig = np.zeros((p1.m, maxTime))   #Gant Diagramm
currentConfig = np.zeros((p1.m, maxTime))
eps = 0.3
idleProb = 0.05   

def learn():
    while(not np.all(nextMaschine[i] == -1 for i in range(0,p1.n))):
        options = getCurrentOptions()
        idle = np.random.random_sample()
        while(len(options) > 0 & idle > idleProb):
            rand = np.random.choise(options)
            best = getMaxQValue(options)
            choose = np.random.choise([rand, best], 1, p=[eps, 1-eps])
            schedule(options[choose])
            options = getCurrentOptions()
        

def getMaxQValue(options):
    bestO = options[0]
    bestValue = 0
    for o in options:
        i = o[0]
        j = o[1]
        mTable = q[j]
        cur = mTable[orderPointer[j]][i] #Q wert zu tupel (i,j)=o
        if(cur > bestValue):
            bestO = o
            bestValue = cur
    return bestO




def getCurrentOptions():
    result = []
    for i in range(0, p1.n):
        j = nextMaschine[i]
        if(j != -1):
            if(belegung[j][i] == 0):
                result.append((i+1,j))
    return result #tupel (auftrag, maschine)

def schedule(t):
    i = t[0]
    j = t[1]
    #update all





#for i in range(100):
#    learn()

#print(getCurrentOptions())

#print(bestConfig)


