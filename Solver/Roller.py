import pandas as pd

class Roller:

    def __init__(self, CenterX, CenterY, Radius, AngularVelocity):
        self.CenterX = CenterX
        self.CenterY = CenterY
        self.Radius = Radius
        self.AngularVelocity = AngularVelocity

    def printRoller(self):
        print("CenterX", self.CenterX)
        print("CenterY", self.CenterY)
        print("Radius", self.Radius)
        print("AngularVelocity", self.AngularVelocity)

    def saveParams(self, address, RollerNumber):
        df = pd.DataFrame(columns=["CenterX", "CenterY", "Radius", "AngularVelocity"])
        df.loc[0, "CenterX"] = self.CenterX
        df.loc[0, "CenterY"] = self.CenterY
        df.loc[0, "Radius"] = self.Radius
        df.loc[0, "AngularVelocity"] = self.AngularVelocity

        df.to_csv(f"{address}/RollerParams{RollerNumber}.csv")
