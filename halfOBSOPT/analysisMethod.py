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
        self.truthState = np.loadtxt("./initRecord/{}/sparseTruthState.txt".format(self.subFolderName))
        if "ourDVar" in self.methodName:
            self.RMSE[-1] = np.nan
            self.meanError[-1] = np.nan

class drawSys:
    def __init__(self, subFolderName):
        self.subFolderName = subFolderName
        self.noiseType, self.noiseScale = subFolderName.split("_")
        self.collectAllResult()

    def collectAllResult(self):
        self.noDA = dataReader(methodName="noDA", subFolderName=self.subFolderName)
        self.ekf = dataReader(methodName="EKF", subFolderName=self.subFolderName)
        self.threeDvar = dataReader(methodName="threeDVar", subFolderName=self.subFolderName)
        self.increThreeDvar = dataReader(methodName="increThreeDVar", subFolderName=self.subFolderName)
        self.fourDvar = dataReader(methodName="fourDVar", subFolderName=self.subFolderName)
        self.increFourDvar = dataReader(methodName="increFourDVar", subFolderName=self.subFolderName)

    def drawErrorInfo(self):
        
        """draw RMSE and Mean Error"""
        gs = gridspec.GridSpec(2, 2)
        figure(figsize=(17, 8), dpi=200)
        ax = subplot(gs[0, :])
        grid(True)
        plot(timeArray, self.ekf.RMSE, label=self.ekf.methodName, linewidth=3, color="#1f77b4")
        plot(timeArray, self.threeDvar.RMSE, label=self.threeDvar.methodName, linewidth=3, color="#ff7f0e")
        plot(timeArray, self.increThreeDvar.RMSE, "--", label=self.increThreeDvar.methodName, linewidth=3, color="#2ca02c")
        plot(timeArray, self.fourDvar.RMSE, label=self.fourDvar.methodName, linewidth=3, color="#d62728")
        plot(timeArray, self.increFourDvar.RMSE, "--", label=self.increFourDvar.methodName, linewidth=3, color="#9467bd")
        plot(timeArray, self.noDA.RMSE, label=self.noDA.methodName, linewidth=3, color="black")
        xlim(np.min(timeArray), np.max(timeArray));ylim(-0.5, 7)
        xlabel("Time", fontsize=12);ylabel("RMSE", fontsize=12)
        legend(fontsize=10, loc='right')
        title("Error Information | {HType} | NoiseType: {NT} | NoiseScale: {NS}"\
              .format(HType=observationOperatorType, NT=self.noiseType, NS=self.noiseScale), 
              fontsize=15)
        #
        ax = subplot(gs[1, 0])
        grid(True)
        plot(timeArray, self.ekf.RMSE, linewidth=3, color="#1f77b4", alpha=0.2)
        plot(timeArray, self.threeDvar.RMSE, linewidth=3, color="#ff7f0e", alpha=0.2)
        plot(timeArray, self.increThreeDvar.RMSE, "--", linewidth=3, color="#2ca02c", alpha=0.2)
        plot(timeArray, self.fourDvar.RMSE, linewidth=3, color="#d62728")
        plot(timeArray, self.increFourDvar.RMSE, linewidth=3, color="#9467bd")
        #plot(timeArray, noDA.RMSE, label=noDA.methodName, linewidth=3, color="black")
        xlim(np.min(timeArray), np.max(timeArray));ylim(-0.0, 1.0)
        xlabel("Time");ylabel("RMSE")

        ax = subplot(gs[1, 1])
        grid(True)
        plot(timeArray, self.ekf.meanError, linewidth=2, color="#1f77b4", alpha=0.2)
        plot(timeArray, self.threeDvar.meanError, linewidth=2, color="#ff7f0e", alpha=0.2)
        plot(timeArray, self.increThreeDvar.meanError, "--", linewidth=2, color="#2ca02c", alpha=0.2)
        plot(timeArray, self.fourDvar.meanError, linewidth=2, color="#d62728")
        plot(timeArray, self.increFourDvar.meanError, linewidth=2, color="#9467bd")
        #plot(timeArray, noDA.meanError, label=noDA.methodName, linewidth=3, color="black")
        xlim(np.min(timeArray), np.max(timeArray));ylim(-0.2, 0.2)
        xlabel("Time");ylabel("Mean Error")
        savefig("RMSE_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg"\
                .format(HType=observationOperatorType, NT=noiseType, NS=self.noiseScale))
        print("RMSE_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg saved"\
              .format(HType=observationOperatorType, NT=noiseType, NS=self.noiseScale))

    def drawMixingSplit(self, method, color):
        gs = gridspec.GridSpec(3, 1)
        figure(figsize=(17, 8), dpi=200)
        ax = subplot(gs[:2])
        grid(True)
        laplaceRMSE, gaussianRMSE = self.splitGaussianLaplace(method=method)
        plot(timeArray, method.RMSE, label=method.methodName, linewidth=3, color=color)
        plot(timeArray, laplaceRMSE, "--", label=method.methodName+"_L", linewidth=3, color=color, alpha=0.5)
        plot(timeArray, gaussianRMSE, ":", label=method.methodName+"_G", linewidth=3, color=color)
        xlim(np.min(timeArray), np.max(timeArray));ylim(-0.5, 7)
        ylabel("RMSE", fontsize=12)
        legend(fontsize=10)
        title("Gaussian & Laplace from Mixing | {HType} | NoiseType: {NT} | NoiseScale: {NS}"\
              .format(HType=observationOperatorType, NT=self.noiseType, NS=self.noiseScale), 
              fontsize=15)

        ax = subplot(gs[2])
        grid(True)
        grid(True)
        laplaceRMSE, gaussianRMSE = self.splitGaussianLaplace(method=method)
        plot(timeArray, method.RMSE, label=method.methodName, linewidth=3, color=color)
        plot(timeArray, laplaceRMSE, "--", label=method.methodName+"_L", linewidth=3, color=color, alpha=0.5)
        plot(timeArray, gaussianRMSE, ":", label=method.methodName+"_G", linewidth=3, color=color)
        xlim(np.min(timeArray), np.max(timeArray));ylim(-0.0, 0.75)
        xlabel("Time", fontsize=12); ylabel("RMSE", fontsize=12)
        savefig("MixingSplit_{MN}_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg"\
                .format(MN=method.methodName, HType=observationOperatorType, NT=noiseType, NS=self.noiseScale))
        print("MixingSplit_{MN}_{HType}_NoiseType_{NT}_NoiseScale_{NS}.jpg saved"\
              .format(MN=method.methodName, HType=observationOperatorType, NT=noiseType, NS=self.noiseScale))

    def splitGaussianLaplace(self, method):
        laplaceRMSE = np.sqrt(np.mean((method.analysisState[:, :20] - method.truthState[:, :20])**2, axis=1))
        gaussianRMSE = np.sqrt(np.mean((method.analysisState[:, 20:] - method.truthState[:, 20:])**2, axis=1))
        if "ourDVar" in method.methodName:
            laplaceRMSE[-1] = np.nan
            gaussianRMSE[-1] = np.nan
        return laplaceRMSE, gaussianRMSE

if __name__ == "__main__":
    subFolderName = "Mixing0.5_0.4"
    truthState = np.loadtxt("./initRecord/{}/truthState.txt".format(subFolderName))
    observationState = np.loadtxt("./initRecord/{}/fullObservationState.txt".format(subFolderName))

    drawsys = drawSys(subFolderName=subFolderName)
    #drawsys.drawErrorInfo()
    drawsys.drawMixingSplit(method=drawsys.threeDvar, color="#ff7f0e")
    drawsys.drawMixingSplit(method=drawsys.fourDvar, color="#d62728")
    drawsys.drawMixingSplit(method=drawsys.increFourDvar, color="#9467bd")
    #show()