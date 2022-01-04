import numpy as np
from scipy.integrate import ode
from matplotlib.pyplot import *
from initValueGenerate import Lorenz96
from parameterControl import *
from dataRecorder import RecordCollector
class ExtKalFil:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.loadtxt("{}/initRecord/observationOperator.txt".format(observationOperatorType))

    def forceODE(self, x, force=force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

    # forecasting
    def getForecastState(self, analysisState, startTime):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=startTime)
        forecastState = lorenz.solveODE(endTime=startTime+dT)
        return forecastState

    def getForcastEC(self, analysisState, analysisEC):
        jacobianM = self.getJacobianOfMfromTransition(analysisState)
        forecastEC = jacobianM @ analysisEC @ jacobianM.transpose()
        return forecastEC

    def getJacobianOfForceODE(self, xState):
        """F(t_i)"""
        jacobianOfForceMaxtrix = np.zeros((xState.shape[0], xState.shape[0]), dtype=float)
        x_n = xState
        x_np1 = np.roll(xState, -1) # x_n+1
        x_nm1 = np.roll(xState, +1) # x_n-1
        x_nm2 = np.roll(xState, +2) # x_n-2
        for i in range(len(xState)):
            tempState = np.array([-x_nm1[i], x_np1[i] - x_nm2[i], -1, x_nm1[i]])
            jacobianOfForceMaxtrix[i, :4] = tempState
            jacobianOfForceMaxtrix[i] = np.roll(jacobianOfForceMaxtrix[i], -2+i)
        return jacobianOfForceMaxtrix

    def getJacobianOfMfromTransition(self, analysisState, Nsplit=10):
        tempState = analysisState
        ddT = dT / Nsplit
        M = np.identity(len(tempState))
        for i in range(Nsplit):
            F = self.getJacobianOfForceODE(xState = tempState)
            L_i = np.identity(Ngrid) + ddT * F
            tempState = tempState + ddT * self.forceODE(x=tempState)
            M = L_i @ M
        return M

    # analyzing
    def getAnalysisWeight(self, forecastState, forecastEC, observationEC):
        H = self.obsOperator
        inverseMatrix = np.linalg.inv(H @ forecastEC @ (H.transpose()) + observationEC)
        K = forecastEC @ (H.transpose()) @ (inverseMatrix)
        return K

    def getAnalysisState(self, forecastState, observationState, KalmanGain):
        H = self.obsOperator
        innovation = (observationState - H @ forecastState)
        analysisState = forecastState + (KalmanGain @ innovation)
        return analysisState

    def getAnalysisEC(self, forecastEC, KalmanGain, inflation=1):
        H = self.obsOperator
        analysisEC = (np.identity(Ngrid) - (KalmanGain @ H)) @ forecastEC
        analysisEC = inflation * analysisEC
        return analysisEC

if __name__ == "__main__":
    # states
    xInitAnalysis = np.loadtxt("{}/initRecord/{}/initAnalysisState.txt".format(observationOperatorType, subFolderName))
    xFullObservation = np.loadtxt("{}/initRecord/{}/sparseObservationState.txt".format(observationOperatorType, subFolderName))
    xTruth = np.loadtxt("{}/initRecord/{}/sparseTruthState.txt".format(observationOperatorType, subFolderName))

    # covariance 
    analysisEC = np.loadtxt("{}/initRecord/{}/initEC.txt".format(observationOperatorType, subFolderName))
    observationEC = np.identity(Ngrid) * (noiseScale ** 2)

    # collector
    dataRecorder = RecordCollector(methodName="EKF", noiseType=noiseType)
    if not dataRecorder.checkDirExists():
        dataRecorder.makeDir()

    # initial setup
    ekf = ExtKalFil(xInitAnalysis)
    ekf.analysisState = xInitAnalysis
    ekf.analysisEC = analysisEC
    ekf.observationState = xFullObservation[0]
    ekf.observationEC = observationEC
    ekf.forecastState = np.zeros(Ngrid)
    ekf.forecastEC = np.zeros((Ngrid, Ngrid))
    ekf.RMSE = np.sum((ekf.analysisState - xTruth[0])**2)
    ekf.MeanError = np.mean(ekf.analysisState - xTruth[0])
    dataRecorder.record(ekf, tidx=0)
    print("{:02f}: {:05f}".format(0, ekf.RMSE))

    for tidx, nowT in enumerate(timeArray[:-1]):
        ekf.forecastState = ekf.getForecastState(analysisState=ekf.analysisState, startTime=nowT)
        ekf.forecastEC = ekf.getForcastEC(analysisState=ekf.analysisState, analysisEC=ekf.analysisEC)

        ekf.observationState = xFullObservation[tidx+1]
        ekf.KalmanGain = ekf.getAnalysisWeight(forecastState=ekf.forecastState, \
                                               forecastEC=ekf.forecastEC, \
                                               observationEC=ekf.observationEC)
        ekf.analysisState = ekf.getAnalysisState(forecastState=ekf.forecastState, \
                                                 observationState=ekf.observationState, \
                                                 KalmanGain=ekf.KalmanGain)
        ekf.analysisEC = ekf.getAnalysisEC(forecastEC=ekf.forecastEC, KalmanGain=ekf.KalmanGain, inflation=2)
        ekf.RMSE = np.sqrt(np.mean((ekf.analysisState - xTruth[tidx+1])**2))
        ekf.MeanError = np.mean(ekf.analysisState - xTruth[tidx+1])
        dataRecorder.record(ekf, tidx=tidx+1)
        print("{:02f}: {:05f}".format(nowT+dT, ekf.RMSE))
    dataRecorder.saveToTxt()



