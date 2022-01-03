from parameterControl import *
import pathlib

class RecordCollector:
    def __init__(self, methodName, noiseType):
        self.methodName = methodName
        self.noiseType = noiseType
        self.analysisState = np.zeros((NtimeStep, Ngrid))
        #self.analysisEC    = np.zeros((NtimeStep, Ngrid, Ngrid))
        self.forecastState = np.zeros((NtimeStep, Ngrid))
        #self.forecastEC    = np.zeros((NtimeStep, Ngrid, Ngrid))
        self.observationState = np.zeros((NtimeStep, Ngrid))
        #self.observationEC = np.zeros((NtimeStep, Ngrid, Ngrid))
        self.RMSE          = np.zeros(NtimeStep)
        self.MeanError     = np.zeros(NtimeStep)

    def record(self, DAClass, tidx):
        self.analysisState[tidx] = DAClass.analysisState
        #self.analysisEC[tidx] = DAClass.analysisEC
        self.forecastState[tidx] = DAClass.forecastState
        #self.forecastEC[tidx] = DAClass.forecastEC
        self.observationState[tidx] = DAClass.observationState
        #self.observationEC[tidx] = DAClass.observationEC
        self.RMSE[tidx] = DAClass.RMSE
        self.MeanError[tidx] = DAClass.MeanError

    def checkDirExists(self):
        goalDir = "./{}_record".format(self.methodName)
        return pathlib.Path(goalDir).is_dir()

    def makeDir(self):
        goalDir = "./{}_record".format(self.methodName)
        pathlib.Path(goalDir).mkdir(parents=True, exist_ok=True)

    def saveToTxt(self):
        np.savetxt("{}_record/{}_{}_analysisState.txt".format(self.methodName, self.noiseType, noiseScale), self.analysisState)
        #np.savetxt("{}_record/{}_analysisEC.txt".format(self.methodName, self.noiseType), self.analysisEC)
        np.savetxt("{}_record/{}_{}_forecastState.txt".format(self.methodName, self.noiseType, noiseScale), self.forecastState)
        #np.savetxt("{}_record/{}_forecastEC.txt".format(self.methodName, self.noiseType), self.forecastEC)
        np.savetxt("{}_record/{}_{}_observationState.txt".format(self.methodName, self.noiseType, noiseScale), self.observationState)
        #np.savetxt("{}/{}_observationEC.txt".format(self.methodName, self.noiseType), self.observationEC)
        np.savetxt("{}_record/{}_{}_RMSE.txt".format(self.methodName, self.noiseType, noiseScale), self.RMSE)
        np.savetxt("{}_record/{}_{}_MeanError.txt".format(self.methodName, self.noiseType, noiseScale), self.MeanError)
        print("Save successfully")