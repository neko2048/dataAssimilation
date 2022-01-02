import numpy as np
from scipy.integrate import ode
from matplotlib.pyplot import *
import copy
from initValueGenerate import Lorenz96
from parameterControl import *
#import lorenz96
#from settings import *

class fourDVar:
    def __init__(self, xInitAnalysis):
        self.xInitAnalysis = xInitAnalysis
        self.obsOperator = np.identity(Ngrid)

    def forceODE(self, x, force=force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

    # forecasting
    def getForecastState(self, analysisState, startTime):
        pass

    def getForcastEC(self, analysisState, analysisEC):
        pass

    # analyzing
    def getAnalysisState(self, forecastState, observationState, KalmanGain):
        pass

    def getAnalysisEC(self, forecastEC, KalmanGain, inflation=1):
        pass

if __name__ == "__main__":
    # states
    xInitAnalysis = np.loadtxt("initRecord/xAnalysisInit.txt")
    xFullObservation = np.loadtxt("initRecord/xObservation_{}.txt".format(noiseType))
    print(xFullObservation.shape)
    xTruth = np.loadtxt("initRecord/xTruth.txt")

    # covariance 
    analysisEC = np.loadtxt("./initEC_{}.txt".format(noiseType))
    observationEC = np.identity(Ngrid) * 0.4

    # collector
    dataRecorder = RecordCollector(methodName="EKF", noiseType=noiseType)

    # initial setup
    ekf = ExtKalFil(xInitAnalysis)
    ekf.analysisState = xInitAnalysis
    ekf.analysisEC = analysisEC
    ekf.observationState = xFullObservation[0]
    ekf.observationEC = observationEC
    print("{NT:02f}: {ERROR:05f}".format(NT=0, ERROR=np.sum((ekf.analysisState - ekf.observationState)**2)))

    for i, nowT in enumerate(timeArray[:-1]):
        ekf.forecastState = ekf.getForecastState(analysisState=ekf.analysisState, startTime=nowT)
        ekf.forecastEC = ekf.getForcastEC(analysisState=ekf.analysisState, analysisEC=ekf.analysisEC)

        ekf.observationState = xFullObservation[i+1]
        ekf.KalmanGain = ekf.getAnalysisWeight(forecastState=ekf.forecastState, \
                                               forecastEC=ekf.forecastEC, \
                                               observationEC=ekf.observationEC)
        ekf.analysisState = ekf.getAnalysisState(forecastState=ekf.forecastState, \
                                                 observationState=ekf.observationState, \
                                                 KalmanGain=ekf.KalmanGain)
        ekf.analysisEC = ekf.getAnalysisEC(forecastEC=ekf.forecastEC, KalmanGain=ekf.KalmanGain, inflation=2.0)
        print("{NT:02f}: {ERROR:05f}".format(NT=nowT+dT, ERROR=np.sqrt(np.mean((ekf.analysisState - ekf.observationState)**2))))
