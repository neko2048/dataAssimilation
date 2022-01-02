import numpy as np
from scipy.integrate import ode
from matplotlib.pyplot import *
import copy
from initValueGenerate import Lorenz96
from parameterControl import *
from scipy.optimize import minimize
from dataRecorder import RecordCollector

class threeDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.identity(Ngrid)

    def costFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        H = self.obsOperator
        backgroundCost = (backgroundState - analysisState).transpose() @ np.linalg.inv(backgroundEC) @ (backgroundState - analysisState)

        innovation = observationState - H @ analysisState
        observationCost = (innovation).transpose() @ np.linalg.inv(observationEC) @ (innovation)
        return 0.5 * (backgroundCost + observationCost)

    def gradientOfCostFunction(self, analysisState, backgroundState, observationState, backgroundEC, observationEC):
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (analysisState - backgroundState)

        innovation = observationState - H @ analysisState
        gradientObservationCost = H.transpose() @ np.linalg.inv(observationEC) @ (innovation)
        return gradientBackgroundCost - gradientObservationCost

    # forecast
    def getForecastState(self, analysisState, nowT):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=nowT)
        backgroundState = lorenz.solveODE(endTime=nowT+dT)
        return backgroundState

    # analyzing
    def getAnalysisState(self, backgroundState, observationState, backgroundEC, observationEC):
        analysisState = minimize(self.costFunction, x0=backgroundState, \
                        args = (backgroundState, observationState, backgroundEC, observationEC), \
                        method='CG', jac=self.gradientOfCostFunction).x
        return analysisState

    def getAnalysisEC(self, forecastEC, KalmanGain, inflation=1):
        pass

if __name__ == "__main__":
    # states
    xInitAnalysis = np.loadtxt("initRecord/xAnalysisInit.txt")
    xFullObservation = np.loadtxt("initRecord/xObservation_{}.txt".format(noiseType))
    xTruth = np.loadtxt("initRecord/xTruth.txt")

    # covariance 
    analysisEC = np.loadtxt("initRecord/initEC_{}.txt".format(noiseType))
    observationEC = np.identity(Ngrid) * (noiseScale ** 2)

    # collector
    dataRecorder = RecordCollector(methodName="threeDVar", noiseType=noiseType)

    # initial setup
    threeDvar = threeDVar(xInitAnalysis)
    threeDvar.analysisState = xInitAnalysis
    threeDvar.analysisEC = analysisEC
    threeDvar.backgroundState = xInitAnalysis # presumed
    threeDvar.backgroundEC = analysisEC # presumed
    threeDvar.observationState = xFullObservation[0]
    threeDvar.observationEC = observationEC
    print("{NT:02f}: {ERROR:05f}".format(NT=0, ERROR=np.sum((threeDvar.analysisState - xTruth[0])**2)))

    for tidx, nowT in enumerate(timeArray[:-1]):
        threeDvar.observationState = xFullObservation[tidx+1]

        threeDvar.backgroundState = threeDvar.getForecastState(threeDvar.analysisState, nowT=nowT)
        threeDvar.analysisState = threeDvar.getAnalysisState(backgroundState = threeDvar.backgroundState, \
                                                             observationState = threeDvar.observationState, \
                                                             backgroundEC = threeDvar.backgroundEC, \
                                                             observationEC = threeDvar.observationEC)
        print("{NT:02f}: {ERROR:05f}".format(NT=nowT+dT, ERROR=np.sqrt(np.mean((threeDvar.analysisState - xTruth[tidx+1])**2))))



