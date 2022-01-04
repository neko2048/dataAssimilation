import numpy as np 
from matplotlib.pyplot import *
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
    ekf_L04 = dataReader(methodName="EKF", subFolderName="Laplace_0.4")
    threeDvar_L04 = dataReader(methodName="threeDVar", subFolderName="Laplace_0.4")
    increThreeDvar_L04 = dataReader(methodName="increThreeDVar", subFolderName="Laplace_0.4")
    fourDvar_L04 = dataReader(methodName="fourDVar", subFolderName="Laplace_0.4")
    increFourDvar_L04 = dataReader(methodName="increFourDVar", subFolderName="Laplace_0.4")

    plot(timeArray, ekf_L04.RMSE, label=ekf_L04.methodName, color="#1f77b4")
    plot(timeArray, threeDvar_L04.RMSE, label=threeDvar_L04.methodName, color="#ff7f0e")
    plot(timeArray, increThreeDvar_L04.RMSE, label=increThreeDvar_L04.methodName, color="#2ca02c")
    plot(timeArray, fourDvar_L04.RMSE, label=fourDvar_L04.methodName, color="#d62728")
    plot(timeArray, increFourDvar_L04.RMSE, label=increFourDvar_L04.methodName, color="#9467bd")
    
    plot(timeArray, ekf_L04.meanError, "--", color="#1f77b4")
    plot(timeArray, threeDvar_L04.meanError, "--", color="#ff7f0e")
    plot(timeArray, increThreeDvar_L04.meanError, "--", color="#2ca02c")
    plot(timeArray, fourDvar_L04.meanError, "--", color="#d62728")
    plot(timeArray, increFourDvar_L04.meanError, "--", color="#9467bd")

    legend()
    ylim(-0.5, 2)
    show()