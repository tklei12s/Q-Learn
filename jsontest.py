import pandas as pd
import json


class myProblem:
    with open(r'C:\Users\Tom\Desktop\example.json') as f:
        data = json.load(f)
    id = data["id"]
    reihenfolge = data["reihenfolge"]
    bearbeitungszeiten = data["bearbeitungszeiten"]
    zielfunktion = data["zielfunktion"]
    if(zielfunktion == "Lmax"):
        liefertermin = data["liefertermin"]

p1 = myProblem()

print(p1.reihenfolge)

