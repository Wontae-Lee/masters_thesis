import numpy as np
import pandas as pd


class Mesh:
    # ny: The number of y-mesh
    # nx: The number of x-mesh
    # start_x: The start point of x-mesh
    # end_x: The end point of y-mesh
    # start_y: The start point of y-mesh
    # end_y: The end point of y-mesh

    def __init__(self, start_x, end_x, start_y, end_y ,nx =100, ny =100):

        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.nx = nx
        self.ny = ny

    def printMesh(self):

        print(f'The start point of x-mesh: {self.start_x}')
        print(f'The end point of x-mesh: {self.end_x}')
        print(f'The start point of y-mesh: {self.start_y}')
        print(f'The end point of y-mesh: {self.end_y}')

    def saveParams(self, address):

        df = pd.DataFrame(columns=["start_x","end_x","start_y","end_y"])
        df.loc[0, "start_x"] = self.start_x
        df.loc[0, "end_x"] = self.end_x
        df.loc[0, "start_y"] = self.start_y
        df.loc[0, "end_y"] = self.end_y

        df.to_csv(f"{address}/MeshParams.csv")



