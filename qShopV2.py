import pandas as pd
import json
import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import sys
from time import time



class Problem:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, sys.argv[1])
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


problemObj = Problem()



#-----------------------

#fuer den aktuellen State

maxZeit = np.matrix.sum(np.matrix(problemObj.bearbeitungszeiten))   #Maximale Zeit, alle Jobs hintereinander
belegung = np.zeros((problemObj.m, problemObj.n), dtype=int)       #null = Maschine frei, nicht null = Zeit bis Maschine frei
zuBearbeiten = deepcopy(problemObj.bearbeitungszeiten)    #null = ist bearbeitet, nicht null muss bearbeitet werden
nextMaschinePointer = np.zeros((problemObj.n), dtype=int)   #Pointer auf den aktuellen Wert in der Reihenfolge Matrix
nextMaschine = deepcopy(problemObj.reihenfolge[0])        #nextMaschine[i] gibt naechste Machine fuer Aufrtrag i. Falls -1 ist, ist Aufrag i fertig
working = []                            #Liste aller aktuell laufenden Prozesse (Auftrag, Maschine Tupel)
maxRandQValue =  15                     #Q Table zufaellig von 0 bis maxRandQValue
orderPointer = np.zeros(problemObj.m, dtype=int)           #Zeigt auf die Stelle in der Q tabelle der Machinen des index, der als naechsts dran ist
q = np.zeros((problemObj.m, problemObj.n, problemObj.n), dtype=float)       #[Maschine][orderPointer][Job]
#q = np.random.randint(low=0, high=maxRandQValue, size=(problemObj.m, problemObj.n, problemObj.n))    
blocked = np.zeros(problemObj.m, dtype=int)
countAlpha = np.zeros((problemObj.m, problemObj.n, problemObj.n), dtype=int)
bestTime = maxZeit  #aktuell beste zeit
bestConfig = np.zeros((problemObj.m, maxZeit), dtype=int)   #Gant Diagramm
currentConfig = np.zeros((problemObj.m, maxZeit), dtype=int)
meanTimepermachine = [np.mean(problemObj.bearbeitungszeiten[i]) for i in range(0, problemObj.n)]
bestPerMachine = np.ones((problemObj.m), dtype=float)*maxZeit
eps = float(sys.argv[3])
gamma = float(sys.argv[5])
curTime = 0     #aktueller Zeitpunkt
bestChanged = 0 #wie viele Iterartionen ist der beste Wert nicht besser geworden
startTime = time()
finishTime = time()


def learn():
    while(np.matrix.sum(np.matrix(zuBearbeiten)) != 0): #solange Problem nicht geloest
        options = getCurrentOptions()   #moegliche Maschinen/Job Tupel berechnen
        while(len(options) > 0):    #solange noch moegliche Job/Maschinen Tupel vorhanden
            #entscheide ob zufaellig oder Q wert fuer auswahl des tupels
            randNum = np.random.randint(low=0, high=len(options), size=1)[0]
            rand = options[randNum]
            best = getMaxQValue(options)
            chooseNum = np.random.choice(a=[0, 1], size=1, p=[1-eps, eps])
            if chooseNum == 1:
                choose = rand
            else:
                choose = best
            #lege job auf maschine
            updateQ(choose)
            working.append(choose)
            i = choose[0]
            j = choose[1]
            belegung[j][i] = problemObj.bearbeitungszeiten[j][i]
            blocked[j] = 1
            orderPointer[j] += 1
            options = getCurrentOptions()
        updateAll()
    updateAll()
    resetAll()


        
#Gibt bestes Tupel aus Optionen basierend auf der Q Tabelle zurueck
def getMaxQValue(options):
    bestO = options[0]
    bestValue = -1
    for o in options:
        i = o[0]
        j = o[1]
        if(orderPointer[j] != problemObj.n):
            cur = q[j][orderPointer[j]][i] #Q wert zu tupel (i,j)=o
            if(cur > bestValue):
                bestO = o
                bestValue = cur
    return bestO

#berechnet aktuell moegliche Jobs/Maschinen Tupel
def getCurrentOptions():
    result = []
    for i in range(0, problemObj.n):
        j = nextMaschine[i]
        if(j != -1 and blocked[j] == 0):
            if(belegung[j][i] == 0):
                result.append((i,j))
    return result #tupel (auftrag, maschine)
    
#updatet alle Variablen jede Zeiteinheit
def updateAll():
    global curTime, bestConfig, bestTime, working, zuBearbeiten, bestChanged

    removeList = []
    if len(working) == 0 and np.matrix.sum(np.matrix(zuBearbeiten)) == 0:
        bestChanged += 1
        if(curTime < bestTime):
            bestChanged = 0
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
            nextMaschinePointer[i] +=1
            if(nextMaschinePointer[i] == problemObj.m):
                nextMaschine[i] = -1
            else:
                nextMaschine[i] = problemObj.reihenfolge[nextMaschinePointer[i]][i]
    for r in removeList:
        working.remove(r)
    curTime += 1

#resetet alle Variablen nach einem Lerndurchlauf
def resetAll():
    global belegung, zuBearbeiten, nextMaschinePointer, nextMaschine, working, orderPointer, countAlpha, currentConfig, curTime
    curTime = 0
    belegung = np.zeros((problemObj.m, problemObj.n), dtype=int)
    zuBearbeiten = deepcopy(problemObj.bearbeitungszeiten)
    nextMaschinePointer = np.zeros((problemObj.n), dtype=int)
    nextMaschine = deepcopy(problemObj.reihenfolge[0])
    working = []
    orderPointer = np.zeros(problemObj.m, dtype=int)
    currentConfig = np.zeros((problemObj.m, maxZeit), dtype=int)


#updatet die Q Tabelle
def updateQ(t):
    i = t[0]
    j = t[1]
    curPos = orderPointer[j]
    countAlpha[j][curPos][i] += 1 #anzhal besuche in diesem zustand/aktions paar
    alphaValue = 1/(1+countAlpha[j][curPos][i]) #alpha berechnen

    remainingTime = sum(zuBearbeiten[j])
    infi = sum(problemObj.bearbeitungszeiten[j])#groesste untere schranke fuer den fertigstellungszeitpunkt

    x = curTime + remainingTime + (problemObj.n-curPos) - infi #schaetzung abweichung vom infimum
    r = 1000/pow(x,2)   #berechnung reward
    if curPos < problemObj.n-1:
        qMax = np.max([q[j][curPos+1][p] for p in range(0, problemObj.n)])
    else:
        qMax = 0

    q[j][curPos][i] = (1-alphaValue) * q[j][curPos][i] + alphaValue*(r+gamma*qMax)

#gant diagramm anzeigen
def printGant():
    print('Q-Table')
    print(np.round(q))
    print('Zielwert')
    print(bestTime)
    gant = normGant()

    f, ax = plt.subplots(figsize=(11, 7))
    ax = sns.heatmap(data=gant, yticklabels=range(0,problemObj.n), cbar_kws={'ticks': range(0,problemObj.n+1), 'label': 'Jobs'},cmap=sns.cubehelix_palette(n_colors=problemObj.n+1, start=0, rot=0.8, dark=0.1, light=1, reverse=False))
    ax.axhline(y=0, color='k',linewidth=5)
    ax.axhline(y=problemObj.n, color='k',linewidth=5)
    ax.axvline(x=0, color='k',linewidth=5)
    ax.axvline(x=bestTime, color='k',linewidth=5)
    plt.ylabel('Maschine')
    plt.xlabel('Zeiteinheiten')
    plt.show()

#gant diagramm auf bestTime kuerzen
def normGant():
    newBestConfig = np.zeros((problemObj.m, bestTime), dtype=int)
    for i in range(0, problemObj.m):
        for j in range(0, bestTime):
            newBestConfig[i][j] = bestConfig[i][j]
    return newBestConfig

#Json ausgeben
def jsonOut():
    gant = normGant()
    processTime = str(finishTime - startTime)
    processTime=processTime[:-11]
    schedule = np.zeros((problemObj.m, problemObj.n, 2), dtype=int)

    for j in range(0, problemObj.m):
        curJob = 0
        schedulePos = 0
        for i in range (0, bestTime):
            if gant[j][i] != curJob:
                curJob = gant[j][i]
                if curJob != 0:
                    schedule[j][schedulePos][0] = curJob
                    schedule[j][schedulePos][1] = i
                    schedulePos += 1
    
    scheduleList = schedule.tolist()
    data = {
        'name' : problemObj.name,
        'kommentar' : problemObj.kommentar,
        'zielfunktion' : problemObj.zielfunktion,
        'zielwert' : bestTime,
        'schedule' : scheduleList,
        'Laufzeit' : processTime+" Sekunden"
    }

    solName = problemObj.name + 'Sol.json'
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname,solName)

    if(os.path.isfile(filename)):
        os.remove(filename)
    with open(filename, 'x') as f:
        json.dump(data,f)



def start(maxIter):
    startTime = time() 
    while(bestChanged<maxIter):
        learn()
    finishTime = time()
    printGant()
    jsonOut()
    
   
start(int(sys.argv[7]))

