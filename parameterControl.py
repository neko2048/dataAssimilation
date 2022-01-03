import numpy as np
# ========== input parameters
Ngrid = 40
force = 8.
timeLength = 10
dT = 0.05
intensedT = 0.01
NwindowSample = int(dT/intensedT) # used in 4DVar
dx = 1
saveOpt = True
noiseType = "Gaussian"
noiseScale = 0.4
# ========== spin-up settings
initPerturb = 0.1
initSpingUpTime = 100.
# ========== time control
timeArray = np.arange(0, timeLength+dT, dT)
NtimeStep = len(timeArray)
intenseTimeArray = np.arange(0, timeLength+intensedT, intensedT)
intesneNtimeStep = len(intenseTimeArray)

