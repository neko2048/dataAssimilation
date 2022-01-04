import numpy as np 
from matplotlib.pyplot import *
import sys
sys.path.append("../")
from parameterControl import *

class dataReader:
    def __init__(self, methodName, subFolderName):
        self.methodName = methodName
        self.subFolderName = subFolderName
        self.readData()

    def readData(self):
        subFileDir = "./{MN}_record/{SFN}_"\
                       .format(MN=self.methodName, SFN=self.subFolderName)
        self.analysisState = np.loadtxt(subFileDir+"analysisState.txt")
        self.observationState = np.loadtxt(subFileDir+"observationState.txt")
        self.forecastState = np.loadtxt(subFileDir+"forecastState.txt")
        self.meanError = np.loadtxt(subFileDir+"MeanError.txt")
        self.RMSE = np.loadtxt(subFileDir+"RMSE.txt")
        if "ourDVar" in self.methodName:
            self.RMSE[-1] = np.nan
            self.meanError[-1] = np.nan

if __name__ == "__main__":
    subFolderName = "Gaussian_0.4"
    ekf = dataReader(methodName="EKF", subFolderName=subFolderName)
    threeDvar = dataReader(methodName="threeDVar", subFolderName=subFolderName)
    increThreeDvar = dataReader(methodName="increThreeDVar", subFolderName=subFolderName)
    fourDvar = dataReader(methodName="fourDVar", subFolderName=subFolderName)
    increFourDvar = dataReader(methodName="increFourDVar", subFolderName=subFolderName)
    truthState = np.loadtxt("./initRecord/{}/truthState.txt".format(subFolderName))
    observationState = np.loadtxt("./initRecord/{}/fullObservationState.txt".format(subFolderName))

    #figure(figsize=(16, 8))
    #grid(True)
    #plot(timeArray, ekf.RMSE, label=ekf.methodName, color="#1f77b4")
    #plot(timeArray, threeDvar.RMSE, label=threeDvar.methodName, color="#ff7f0e")
    #plot(timeArray, increThreeDvar.RMSE, label=increThreeDvar.methodName, color="#2ca02c")
    #plot(timeArray, fourDvar.RMSE, label=fourDvar.methodName, color="#d62728")
    #plot(timeArray, increFourDvar.RMSE, label=increFourDvar.methodName, color="#9467bd")
    #
    #plot(timeArray, ekf.meanError, "--", color="#1f77b4")
    #plot(timeArray, threeDvar.meanError, "--", color="#ff7f0e")
    #plot(timeArray, increThreeDvar.meanError, "--", color="#2ca02c")
    #plot(timeArray, fourDvar.meanError, "--", color="#d62728")
    #plot(timeArray, increFourDvar.meanError, "--", color="#9467bd")
    #xlabel("Time")
    #ylabel("Error")
    #noiseType, noiseScale = subFolderName.split("_")
    #title("RMSE (Solid) & Mean Error (Dash) | Half Observation Operator | NoiseType: {NT} | NoiseScale: {NS}"\
    #      .format(NT=noiseType, NS=noiseScale))
    #legend()
    #ylim(-0.5, 2)
    #xlim(0, 10)

    show()