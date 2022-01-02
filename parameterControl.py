import numpy as np
# ========== input parameters
Ngrid = 40
force = 8.
timeLength = 10
dT = 0.05
dx = 1
saveOpt = False
noiseType = "Gaussian"
noiseScale = 0.4
# ========== spin-up settings
initPerturb = 0.1
initSpingUpTime = 100.
# ========== time control
timeArray = np.arange(0, timeLength+dT, dT)
NtimeStep = len(timeArray)