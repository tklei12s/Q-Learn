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

maxRandQValue = 15                      #Q Table zufällig zwischen 0 und 15
q = np.random.randint(low=0, high=maxRandQValue, size=(p1.m, p1.n, p1.n))    #[Maschine][orderPointer][Job]



class Sheduler:
    maxTime = np.matrix.sum(np.matrix(p1.bearbeitungszeiten)) 
    bestTime = maxTime
    bestConfig = np.zeros((p1.m, maxTime), dtype=int)   #Gant Diagramm
    belegung = np.zeros((p1.m, p1.n), dtype=int)       #null = Maschine frei, nicht null = Zeit bis Maschine frei
    zuBearbeiten = deepcopy(p1.bearbeitungszeiten)    #null = ist bearbeitet, nicht null muss bearbeitet werden
    nextMachinePointer = np.zeros((p1.n), dtype=int)   #Pointer auf den aktuellen Wert in der Reihenfolge Matrix
    nextMaschine = deepcopy(p1.reihenfolge[0])        #nextMaschine[i] gibt nächste Machine für Aufrtrag i. Falls -1 ist, ist Aufrag i fertig
    working = []                            #Liste aller aktuell laufenden Prozesse (Auftrag, Maschine Tupel)
    orderPointer = np.zeros(p1.m, dtype=int)           #Zeigt auf die Stelle in der Q tabelle der Machinen des index, der als nächsts dran ist
    blocked = np.zeros(p1.m, dtype=int)
    countAlpha = np.zeros((p1.m, p1.n, p1.n), dtype=int)
    currentConfig = np.zeros((p1.m, maxTime), dtype=int)
    meanTimepermachine = [np.mean(p1.bearbeitungszeiten[i]) for i in range(0, p1.n)]
    bestPerMachine = np.ones((p1.m), dtype=float)*99999
    eps = 0.3
    idleProb = 0.05
    gamma = 0.9
    curTime = 0


    def learn(self):
        while(np.matrix.sum(np.matrix(self.zuBearbeiten)) != 0):
            options = self.getCurrentOptions()
            idle = np.random.random_sample()
            while(len(options) > 0 and idle > self.idleProb):
                idle = np.random.random_sample()
                randNum = np.random.randint(low=0, high=len(options), size=1)[0]
                rand = options[randNum]
                best = self.getMinQValue(options)
                chooseNum = np.random.choice(a=[0, 1], size=1, p=[self.eps, 1-self.eps])
                if chooseNum == 1:
                    choose = rand
                else:
                    choose = best
                self.updateQ(choose)
                self.working.append(choose)
                i = choose[0]
                j = choose[1]
                self.belegung[j][i] = p1.bearbeitungszeiten[j][i]
                self.blocked[j] = 1
                self.orderPointer[j] += 1
                options = self.getCurrentOptions()
            self.updateAll()
        self.updateAll()
        self.resetAll()


            

    def getMinQValue(self, options):
        bestO = options[0]
        bestValue = 99999
        for o in options:
            i = o[0]
            j = o[1]
            if(self.orderPointer[j] != p1.n):
                cur = q[j][self.orderPointer[j]][i] #Q wert zu tupel (i,j)=o
                if(cur < bestValue):
                    bestO = o
                    bestValue = cur
        return bestO


    def getCurrentOptions(self):
        result = []
        for i in range(0, p1.n):
            j = self.nextMaschine[i]
            if(j != -1 and self.blocked[j] == 0):
                if(self.belegung[j][i] == 0):
                    result.append((i,j))
        return result #tupel (auftrag, maschine)
        

    def updateAll(self):
        global curTime, bestConfig, bestTime, working, zuBearbeiten

        removeList = []
        if len(self.working) == 0 and np.matrix.sum(np.matrix(self.zuBearbeiten)) == 0:
            if(self.curTime < self.bestTime):
                self.bestTime = self.curTime
                bestConfig = self.currentConfig
        for t in self.working:
            i = t[0]
            j = t[1]
            self.currentConfig[j][self.curTime] = i+1
            self.belegung[j][i] -= 1
            self.zuBearbeiten[j][i] -= 1
            if self.belegung[j][i] == 0:     #fertig geworden
                self.blocked[j] = 0
                removeList.append(t)
                self.nextMachinePointer[i] +=1
                if(self.nextMachinePointer[i] == p1.m):
                    self.nextMaschine[i] = -1
                else:
                    self.nextMaschine[i] = p1.reihenfolge[self.nextMachinePointer[i]][i]

        for r in removeList:
            self.working.remove(r)
            
        self.curTime += 1

    def resetAll(self):
        global belegung, zuBearbeiten, nextMachinePointer, nextMaschine, working, orderPointer, countAlpha, currentConfig, curTime
        self.curTime = 0
        self.belegung = np.zeros((p1.m, p1.n), dtype=int)
        self.zuBearbeiten = deepcopy(p1.bearbeitungszeiten)
        self.nextMachinePointer = np.zeros((p1.n), dtype=int)
        self.nextMaschine = deepcopy(p1.reihenfolge[0])
        self.working = []
        self.orderPointer = np.zeros(p1.m, dtype=int)
        self.countAlpha = np.zeros((p1.m, p1.n, p1.n), dtype=int)
        self.currentConfig = np.zeros((p1.m, self.maxTime), dtype=int)



    def updateQ(self, t):
        i = t[0]
        j = t[1]
        curPos = self.orderPointer[j]
        self.countAlpha[j][curPos][i] += 1
        alphaValue = 1/(1+self.countAlpha[j][curPos][i])

        if(self.maschineFertig(j, i)): #wenn i der letzte Job auf j ist, dann belohnung = fertigstellungszeitpunkt von i, also damit Cmax fuer Maschine j
            if (self.curTime + p1.bearbeitungszeiten[j][i]) < self.bestPerMachine[j]:
                self.bestPerMachine[j] = self.curTime+p1.bearbeitungszeiten[j][i]

        aproxFinish = (p1.n-curPos)*self.meanTimepermachine[j]+self.curTime
        if aproxFinish>self.bestPerMachine[j]:
            r = aproxFinish-self.bestPerMachine[j]
        else:
            r = 0
        #else:
        #    r = curTime + p1.bearbeitungszeiten[j][i]
        if curPos < p1.n-1:
            qMin = np.min([q[j][curPos+1][p] for p in range(0, p1.n)])
        else:
            qMin = 0
        q[j][curPos][i] = (1-alphaValue) * q[j][curPos][i] + alphaValue * (r + self.gamma * qMin)

    def maschineFertig(self, j, i):
        for p in range(0, p1.n):
            if p == i:
                continue
            if self.zuBearbeiten[j][p] != 0:
                return False
        return True




def printGant(thread):
    print(q)
    newBestConfig = np.zeros((p1.m, thread.bestTime), dtype=int)
    for i in range(0, p1.m):
        for j in range(0, thread.bestTime):
            newBestConfig[i][j] = thread.bestConfig[i][j]
    print(newBestConfig)
    print(thread.bestTime)
    sns.heatmap(newBestConfig)
    plt.show()
    

def threadRun(s):
    for i in range(100):
        s.learn()


def run(threadCunt):


    s1 = Sheduler()
    s2 = Sheduler()
    x1 = threading.Thread(target=s1.learn)
    x1.start()

    printGant(s1)
    
    #for i in range(0, threadCunt):
        #s = Sheduler()
        #x = threading.Thread(target=threadRun, args=(s,))
        #x.start()


run(1)

printGant()


