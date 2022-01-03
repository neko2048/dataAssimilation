from initValueGenerate import Lorenz96
from parameterControl import *


initValue = np.loadtxt("initRecord/Gaussian/initAnalysisState.txt")
lorenz = Lorenz96(initValue)
lorenz.solver = lorenz.getODESolver()
y1 = lorenz.solveODE(endTime=0.05)

