import pandas as pd
import json
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import os
import threading
import time


class myProblem:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'buch.json')
    with open(filename) as f:
        data = json.load(f)
    name = data["name"]
    kommentar = data["kommentar"]
    reihenfolge = data["reihenfolge"] #[Maschine][Job]
    bearbeitungszeiten = data["bearbeitungszeiten"] #[Maschine][Job]
    zielfunktion = data["zielfunktion"]
    if(zielfunktion == "Lmax"):
        liefertermin = data["liefertermin"]
    n = len(reihenfolge[0])
    m = len(reihenfolge)


p1 = myProblem()



#-----------------------

#für den aktuellen State

maxTime = np.matrix.sum(np.matrix(p1.bearbeitungszeiten))
belegung = np.zeros((p1.m, p1.n), dtype=int)       #null = Maschine frei, nicht null = Zeit bis Maschine frei
zuBearbeiten = deepcopy(p1.bearbeitungszeiten)    #null = ist bearbeitet, nicht null muss bearbeitet werden
nextMachinePointer = np.zeros((p1.n), dtype=int)   #Pointer auf den aktuellen Wert in der Reihenfolge Matrix
nextMaschine = deepcopy(p1.reihenfolge[0])        #nextMaschine[i] gibt nächste Machine für Aufrtrag i. Falls -1 ist, ist Aufrag i fertig
working = []                            #Liste aller aktuell laufenden Prozesse (Auftrag, Maschine Tupel)
maxRandQValue = 15                      #Q Table zufällig zwischen 0 und 10
orderPointer = np.zeros(p1.m, dtype=int)           #Zeigt auf die Stelle in der Q tabelle der Machinen des index, der als nächsts dran ist
q = np.random.randint(low=0, high=maxRandQValue, size=(p1.m, p1.n, p1.n))    #[Maschine][orderPointer][Job]
blocked = np.zeros(p1.m, dtype=int)
countAlpha = np.zeros((p1.m, p1.n, p1.n), dtype=int)
bestTime = maxTime
bestConfig = np.zeros((p1.m, maxTime), dtype=int)   #Gant Diagramm
currentConfig = np.zeros((p1.m, maxTime), dtype=int)
meanTimepermachine = [np.mean(p1.bearbeitungszeiten[i]) for i in range(0, p1.n)]
bestPerMachine = np.ones((p1.m), dtype=float)*99999
eps = 0.3
idleProb = 0.05
gamma = 0.9
curTime = 0

def learn():
    while(np.matrix.sum(np.matrix(zuBearbeiten)) != 0):
        options = getCurrentOptions()
        idle = np.random.random_sample()
        while(len(options) > 0 and idle > idleProb):
            idle = np.random.random_sample()
            randNum = np.random.randint(low=0, high=len(options), size=1)[0]
            rand = options[randNum]
            best = getMinQValue(options)
            chooseNum = np.random.choice(a=[0, 1], size=1, p=[eps, 1-eps])
            if chooseNum == 1:
                choose = rand
            else:
                choose = best
            updateQ(choose)
            working.append(choose)
            i = choose[0]
            j = choose[1]
            belegung[j][i] = p1.bearbeitungszeiten[j][i]
            blocked[j] = 1
            orderPointer[j] += 1
            options = getCurrentOptions()
        updateAll()
    updateAll()
    resetAll()


        

def getMinQValue(options):
    bestO = options[0]
    bestValue = 99999
    for o in options:
        i = o[0]
        j = o[1]
        if(orderPointer[j] != p1.n):
            cur = q[j][orderPointer[j]][i] #Q wert zu tupel (i,j)=o
            if(cur < bestValue):
                bestO = o
                bestValue = cur
    return bestO


def getCurrentOptions():
    result = []
    for i in range(0, p1.n):
        j = nextMaschine[i]
        if(j != -1 and blocked[j] == 0):
            if(belegung[j][i] == 0):
                result.append((i,j))
    return result #tupel (auftrag, maschine)
    

def updateAll():
    global curTime, bestConfig, bestTime, working, zuBearbeiten

    removeList = []
    if len(working) == 0 and np.matrix.sum(np.matrix(zuBearbeiten)) == 0:
        if(curTime < bestTime):
            bestTime = curTime
            bestConfig = currentConfig
    for t in working:
        i = t[0]
        j = t[1]
        currentConfig[j][curTime] = i+1
        belegung[j][i] -= 1
        zuBearbeiten[j][i] -= 1
        if belegung[j][i] == 0:     #fertig geworden
            blocked[j] = 0
            removeList.append(t)
            nextMachinePointer[i] +=1
            if(nextMachinePointer[i] == p1.m):
                nextMaschine[i] = -1
            else:
                nextMaschine[i] = p1.reihenfolge[nextMachinePointer[i]][i]

    for r in removeList:
        working.remove(r)
        
    curTime += 1

def resetAll():
    global belegung, zuBearbeiten, nextMachinePointer, nextMaschine, working, orderPointer, countAlpha, currentConfig, curTime
    curTime = 0
    belegung = np.zeros((p1.m, p1.n), dtype=int)
    zuBearbeiten = deepcopy(p1.bearbeitungszeiten)
    nextMachinePointer = np.zeros((p1.n), dtype=int)
    nextMaschine = deepcopy(p1.reihenfolge[0])
    working = []
    orderPointer = np.zeros(p1.m, dtype=int)
    countAlpha = np.zeros((p1.m, p1.n, p1.n), dtype=int)
    currentConfig = np.zeros((p1.m, maxTime), dtype=int)



def updateQ(t):
    i = t[0]
    j = t[1]
    curPos = orderPointer[j]
    countAlpha[j][curPos][i] += 1
    alphaValue = 1/(1+countAlpha[j][curPos][i])

    if(maschineFertig(j, i)): #wenn i der letzte Job auf j ist, dann belohnung = fertigstellungszeitpunkt von i, also damit Cmax fuer Maschine j
        if (curTime + p1.bearbeitungszeiten[j][i]) < bestPerMachine[j]:
            bestPerMachine[j] = curTime+p1.bearbeitungszeiten[j][i]

    aproxFinish = (p1.n-curPos)*meanTimepermachine[j]+curTime
    if aproxFinish>bestPerMachine[j]:
        r = aproxFinish-bestPerMachine[j]
    else:
        r = 0
    #else:
    #    r = curTime + p1.bearbeitungszeiten[j][i]
    if curPos < p1.n-1:
        qMin = np.min([q[j][curPos+1][p] for p in range(0, p1.n)])
    else:
        qMin = 0
    q[j][curPos][i] = (1-alphaValue) * q[j][curPos][i] + alphaValue * (r + gamma * qMin)

def maschineFertig(j, i):
    for p in range(0, p1.n):
        if p == i:
            continue
        if zuBearbeiten[j][p] != 0:
            return False
    return True

def printGant():
    print(q)
    newBestConfig = np.zeros((p1.m, bestTime), dtype=int)
    for i in range(0, p1.m):
        for j in range(0, bestTime):
            newBestConfig[i][j] = bestConfig[i][j]
    print(newBestConfig)
    print(bestTime)
    sns.heatmap(newBestConfig)
    plt.show()
    

def run(threadCunt):
    for i in range(0, threadCunt):
        x1 = threading.Thread(target=learn)
        x1.start()




run(1000)

printGant()


