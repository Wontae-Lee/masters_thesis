import pandas as pd


class Time:
    def __init__(self, start_time, end_time, time_step):
        self.Tstart = start_time
        self.Tend = end_time
        self.TimeStep = time_step
        self.dt = (end_time - start_time) / time_step

    def printTime(self):
        print(f'start time = {self.Tstart}')
        print(f'end time = {self.Tend}')
        print(f'time step = {self.TimeStep}')
        print(f'dt = {self.dt}')

    def saveParams(self, address):
        df = pd.DataFrame(columns=["Tstart", "Tend", "TimeStep"])
        df.loc[0, "Tstart"] = self.Tstart
        df.loc[0, "Tend"] = self.Tend
        df.loc[0, "TimeStep"] = self.TimeStep

        df.to_csv(f"{address}/TimeParams.csv")

