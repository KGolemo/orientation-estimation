import numpy as np
from typing import List
import csv


class Sample:
    def __init__(self, timestamp=0, gyroX=0, gyroY=0, gyroZ=0, accX=0, accY=0, accZ=0, magX=0, magY=0, magZ=0) -> None:
        self.timestamp = timestamp

        self.gyroX = gyroX
        self.gyroY = gyroY
        self.gyroZ = gyroZ

        self.accX = accX
        self.accY = accY
        self.accZ = accZ

        self.magX = magX
        self.magY = magY
        self.magZ = magZ
        pass

    @property
    def gyroList(self):
        return[self.gyroX, self.gyroY, self.gyroZ]
        pass

    @property
    def accList(self):
        return[self.accX, self.accY, self.accZ]
        pass

    @property
    def magList(self):
        return[self.magX, self.magY, self.magZ]
        pass

    def __repr__(self) -> str:
        return(f"IMU Sample | Gyro | {self.gyroList} | Acc | {self.accList} | mag | {self.magList}")
        pass


class DataLoader:
    def __init__(self, file) -> None:
        self.file = file
        self.samples : List(Sample) = [] 
        pass

    def loadSamples(self):

        with open(self.file) as file:
            lines = file.readlines()

            self.columnHeader = lines[1]
            lines = lines[2:]

            csv_lines = csv.reader(lines, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
            
            for s in csv_lines:
                self.samples.append(Sample(s[0],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11]))
        pass

    def printSamples(self):
        print(self.columnHeader)
        for s in self.samples:
                print(s)
        pass

    def getTimestamps(self):
        timestamps = []
        s: Sample
        for s in self.samples:
            timestamps.append(s.timestamp)
        return timestamps
        pass

    def getSamplingPeriod(self):
        timestamps = self.getTimestamps()
        return np.round(np.median(np.diff(timestamps)), 0) * 10**(-6)

    def getGyroSamples(self):
        gyro = []
        s: Sample
        for s in self.samples:
            gyro.append(s.gyroList)
        return gyro
        pass

    def getAccSamples(self):
        accel = []
        s: Sample
        for s in self.samples:
            accel.append(s.accList)
        return accel
        pass

    def getMagSamples(self):
        mag = []
        s: Sample
        for s in self.samples:
            mag.append(s.magList)
        return mag
        pass