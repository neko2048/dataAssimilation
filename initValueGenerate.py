import numpy as np
from scipy.integrate import ode
from matplotlib.pyplot import *
import copy
from parameterControl import *

class Lorenz96:
    def __init__(self, initValue):
        self.initValue = initValue
        self.initSpingUpTime = initSpingUpTime
        self.force = force

    def forceODE(self, time, x, force):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + force

    def getODESolver(self, initTime=0., solve_method='dopri5', Nstep=10000):
        if Nstep:
            solver = ode(self.forceODE).set_integrator(name=solve_method, nsteps=Nstep)
        else:
            solver = ode(self.forceODE).set_integrator(name=solve_method)
        solver.set_initial_value(self.initValue, t=initTime).set_f_params(self.force)
        return solver

    def solveODE(self, endTime):
        self.solver.integrate(endTime)
        xSolution = np.array(self.solver.y, dtype="f8")
        return xSolution

class dataGenerator:
    def __init__(self):
        self.Ngrid = Ngrid
        self.initPerturb = initPerturb
        self.initSpingUpTime = initSpingUpTime
        self.NtimeStep = NtimeStep

    def setupLorenz96(self, Nstep):
        self.xInit = self.initPerturb * np.random.randn(self.Ngrid)
        self.lorenz96 = Lorenz96(self.xInit)
        self.lorenz96.solver = self.lorenz96.getODESolver(Nstep=Nstep)

    def getInitValue(self):
        self.setupLorenz96(Nstep=10000)
        xInit = self.lorenz96.solveODE(endTime=self.initSpingUpTime)
        return xInit

    def getSeriesTruth(self):
        xTruth = self.getInitValue() # initial truth
        self.lorenz96 = Lorenz96(xTruth)
        solver = self.lorenz96.getODESolver()
        nowTimeStep = 1
        while solver.successful() and nowTimeStep <= self.NtimeStep-1: # exclude zero
            solver.integrate(solver.t + dT)
            xTruth = np.vstack([xTruth, [solver.y]])
            if nowTimeStep % 50 == 0:
                print("Current TimeStep: {NTS:03d} | Time: {ST}".format(NTS=nowTimeStep, ST=round(solver.t, 5)))
            nowTimeStep += 1
        return xTruth

    def getSeriesObs(self, loc, scale, noiseType):
        XObservation = copy.deepcopy(self.xTruth)
        if noiseType == "Gaussian":
            for i in range(self.NtimeStep):
                noise = np.random.normal(loc, scale, size=(Ngrid, ))
                XObservation[i] += noise
        elif noiseType == "Laplace":
            for i in range(self.NtimeStep):
                noise = np.random.laplace(loc, scale, size=(Ngrid, ))
                XObservation[i] += noise
        elif noiseType == "None":
                pass
        return XObservation

if __name__ == "__main__":
    # ========== build truth
    stateGenerator = dataGenerator()
    stateGenerator.xTruth = stateGenerator.getSeriesTruth()
    # xTruth shape: (201, 40)

    # ========= build analysis init state
    stateGenerator.xAnalysis = stateGenerator.getInitValue()
    # xAnalysis shape: (40, )

    # ========= build observation
    stateGenerator.xObservation = stateGenerator.getSeriesObs(loc=0.0, scale=noiseScale, noiseType=noiseType)
    # xObservation shape: (201, 40)
    stateGenerator.xObservationEC = np.cov((stateGenerator.xObservation-stateGenerator.xTruth).transpose())

    if saveOpt:
        np.savetxt('initRecord/xTruth.txt', stateGenerator.xTruth)
        np.savetxt('initRecord/xAnalysisInit.txt', stateGenerator.xAnalysis)
        np.savetxt('initRecord/xObservation_{}.txt'.format(noiseType), stateGenerator.xObservation)
        np.savetxt('initRecord/initEC_{}.txt'.format(noiseType), stateGenerator.xObservationEC)
        print("saved successfully in ./initRecord")