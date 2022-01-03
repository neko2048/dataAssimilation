import numpy as np
from scipy.integrate import ode
import copy
from scipy.optimize import minimize

from initValueGenerate import Lorenz96
from parameterControl import *
from dataRecorder import RecordCollector

class fourDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.identity(Ngrid)

    def costFunction(self, analysisIncrement, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        backgroundCost = (analysisIncrement).transpose() @ np.linalg.inv(backgroundEC) @ (analysisIncrement)

        observationCost = (H @ analysisIncrement - innovation).transpose() @ np.linalg.inv(observationEC) @ (H @ analysisIncrement - innovation)
        return 0.5 * (backgroundCost + observationCost)

    def gradientOfCostFunction(self, analysisIncrement, innovation, backgroundEC, observationEC):
        H = self.obsOperator
        gradientBackgroundCost = np.linalg.inv(backgroundEC) @ (analysisIncrement)

        gradientObservationCost = H.transpose() @ np.linalg.inv(observationEC) @ (H @ analysisIncrement - innovation)
        return gradientBackgroundCost + gradientObservationCost

    # forecast
    def getForecastState(self, analysisState, nowT):
        lorenz = Lorenz96(initValue=analysisState)
        lorenz.solver = lorenz.getODESolver(initTime=nowT)
        backgroundState = lorenz.solveODE(endTime=nowT+dT)
        return backgroundState

    # analyzing
    def getAnalysisState(self, backgroundState, observationState, backgroundEC, observationEC, NouterLoop=4):
        guessState = backgroundState
        for outer in range(NouterLoop):
            guessState = self.outerLoop(guessState=guessState, \
                                        observationState = observationState, \
                                        backgroundEC = backgroundEC, 
                                        observationEC = observationEC)
        return guessState

    def outerLoop(self, guessState, observationState, backgroundEC, observationEC, NinnerLoop=5):
        innovation = observationState - self.obsOperator @ guessState
        for inner in range(NinnerLoop):
            analysisIncrement = self.innerLoop(initGuessState = guessState, \
                                               innovation = innovation, \
                                               backgroundEC = backgroundEC, \
                                               observationEC = observationEC)
        guessState += analysisIncrement
        return guessState

    def innerLoop(self, initGuessState, innovation, backgroundEC, observationEC):
        analysisIncrement = minimize(self.costFunction, x0=initGuessState, \
                            args = (innovation, backgroundEC, observationEC), \
                            method='CG', jac=self.gradientOfCostFunction).x
        return analysisIncrement

if __name__ == "__main__":
    # states
    xInitAnalysis = np.loadtxt("initRecord/{}/initAnalysisState.txt".format(noiseType))
    xFullObservation = np.loadtxt("initRecord/{}/sparseObservationState.txt".format(noiseType))
    xTruth = np.loadtxt("initRecord/{}/sparseTruthState.txt".format(noiseType))

    # covariance 
    analysisEC = np.loadtxt("initRecord/{}/initEC.txt".format(noiseType))
    observationEC = np.identity(Ngrid) * (noiseScale ** 2)

    # collector
    dataRecorder = RecordCollector(methodName="fourDVar", noiseType=noiseType)

    # initial setup
    fourDvar = fourDVar(xInitAnalysis)
    fourDvar.analysisState = xInitAnalysis
    fourDvar.analysisEC = analysisEC
    fourDvar.forecastState = xInitAnalysis # presumed
    fourDvar.forecastEC = analysisEC # presumed
    fourDvar.observationState = xFullObservation[0]
    fourDvar.observationEC = observationEC
    fourDvar.RMSE = np.sqrt(np.mean((fourDvar.analysisState - xTruth[0])**2))
    fourDvar.MeanError = np.mean(fourDvar.analysisState - xTruth[0])
    dataRecorder.record(fourDvar, tidx=0)
    print("{:02f}: {:05f}".format(0, fourDvar.RMSE))
    for tidx, nowT in enumerate(timeArray[:-1]):
        fourDvar.observationState = xFullObservation[tidx+1]

        fourDvar.forecastState = fourDvar.getForecastState(fourDvar.analysisState, nowT=nowT)
        fourDvar.analysisState = fourDvar.getAnalysisState(backgroundState = fourDvar.forecastState, \
                                                                       observationState = fourDvar.observationState, \
                                                                       backgroundEC = fourDvar.forecastEC, \
                                                                       observationEC = fourDvar.observationEC)

        fourDvar.RMSE = np.sqrt(np.mean((fourDvar.analysisState - xTruth[tidx+1])**2))
        fourDvar.MeanError = np.mean(fourDvar.analysisState - xTruth[tidx+1])
        dataRecorder.record(fourDvar, tidx=tidx+1)
        print("{:02f}: {:05f}".format(nowT+dT, fourDvar.RMSE))
    dataRecorder.saveToTxt()
