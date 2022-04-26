
import numpy as np
import pandas as pd

class FluidProperties:

    def __init__(self, mass, SmoothingLength, LiquidViscosity, LiquidDropletRadius, restDensity, stiffnessConstant):

        self.m = mass
        self.h = SmoothingLength
        self.nu = LiquidViscosity
        self.rho0 = restDensity
        self.R = LiquidDropletRadius
        self.k = stiffnessConstant

    def print_liquid_property(self):

        print(f'Liquid viscosity = {self.nu}')

    def saveParams(self, address):
        df = pd.DataFrame(columns=["mass", "SmoothingLength", "LiquidViscosity","PolytropicIndex","LiquidDropletRadius","StateConstant"])
        df.loc[0, "mass"] = self.m
        df.loc[0, "SmoothingLength"] = self.h
        df.loc[0, "LiquidViscosity"] = self.nu
        df.loc[0, "restDensity"] = self.rho0
        df.loc[0, "LiquidDropletRadius"] = self.R
        df.loc[0, "stiffnessConstant"] = self.k

        df.to_csv(f"{address}/FluidPropertiesParams.csv")



