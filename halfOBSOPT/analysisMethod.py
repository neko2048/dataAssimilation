import numpy as np 
from matplotlib.pyplot import *
import sys
sys.path.append("../")
from parameterControl import *
import matplotlib.gridspec as gridspec

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
    subFolderName = "Laplace_0.4"
    noDA = dataReader(methodName="noDA", subFolderName=subFolderName)
    ekf = dataReader(methodName="EKF", subFolderName=subFolderName)
    threeDvar = dataReader(methodName="threeDVar", subFolderName=subFolderName)
    increThreeDvar = dataReader(methodName="increThreeDVar", subFolderName=subFolderName)
    fourDvar = dataReader(methodName="fourDVar", subFolderName=subFolderName)
    increFourDvar = dataReader(methodName="increFourDVar", subFolderName=subFolderName)
    truthState = np.loadtxt("./initRecord/{}/truthState.txt".format(subFolderName))
    observationState = np.loadtxt("./initRecord/{}/fullObservationState.txt".format(subFolderName))

    gs = gridspec.GridSpec(2, 2)
    figure(figsize=(17, 8), dpi=200)
    ax = subplot(gs[0, :])
    grid(True)
    plot(timeArray, ekf.RMSE, label=ekf.methodName, linewidth=3, color="#1f77b4")
    plot(timeArray, threeDvar.RMSE, label=threeDvar.methodName, linewidth=3, color="#ff7f0e")
    plot(timeArray, increThreeDvar.RMSE, "--", label=increThreeDvar.methodName, linewidth=3, color="#2ca02c")
    plot(timeArray, fourDvar.RMSE, label=fourDvar.methodName, linewidth=3, color="#d62728")
    plot(timeArray, increFourDvar.RMSE, "--", label=increFourDvar.methodName, linewidth=3, color="#9467bd")
    plot(timeArray, noDA.RMSE, label=noDA.methodName, linewidth=3, color="black")
    xlim(np.min(timeArray), np.max(timeArray));ylim(-0.5, 7)
    xlabel("Time", fontsize=12);ylabel("RMSE Error", fontsize=12)
    legend(fontsize=10, loc='right')
    title("RMSE | {HType} | NoiseType: {NT} | NoiseScale: {NS}"\
          .format(HType=observationOperatorType, NT=noiseType, NS=noiseScale), 
          fontsize=15)
    #
    ax = subplot(gs[1, 0])
    grid(True)
    plot(timeArray, ekf.RMSE, linewidth=3, color="#1f77b4")
    plot(timeArray, threeDvar.RMSE, linewidth=3, color="#ff7f0e")
    plot(timeArray, increThreeDvar.RMSE, "--", linewidth=3, color="#2ca02c")
    plot(timeArray, fourDvar.RMSE, linewidth=3, color="#d62728")
    plot(timeArray, increFourDvar.RMSE, "--", linewidth=3, color="#9467bd")
    plot(timeArray, noDA.RMSE, label=noDA.methodName, linewidth=3, color="black")
    xlim(np.min(timeArray), np.max(timeArray));ylim(-0.5, 7)
    xlabel("Time");ylabel("RMSE Error")
    noiseType, noiseScale = subFolderName.split("_")
    #
    ax = subplot(gs[1, 1])
    grid(True)
    plot(timeArray, ekf.RMSE, linewidth=3, color="#1f77b4")
    plot(timeArray, threeDvar.RMSE, linewidth=3, color="#ff7f0e")
    plot(timeArray, increThreeDvar.RMSE, "--", linewidth=3, color="#2ca02c")
    plot(timeArray, fourDvar.RMSE, linewidth=3, color="#d62728")
    plot(timeArray, increFourDvar.RMSE, "--", linewidth=3, color="#9467bd")
    xlim(np.min(timeArray), np.max(timeArray));ylim(-0.5, 7)
    xlabel("Time");ylabel("RMSE Error")

    savefig("RMSE_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg"\
          .format(HType=observationOperatorType, NT=noiseType, NS=noiseScale))
    print("RMSE_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg saved"\
          .format(HType=observationOperatorType, NT=noiseType, NS=noiseScale))
    #show()