# dataAssimilation
## Data Assimilation Final Project
Description: Data Assmilation (Fall, 2021) Final Project 
DA Methods included: 
* EKF
* 3DVar
* incremental 3DVar
* 4DVar
* incremental 4DVar

## Let's Make the Algorithms Work
### 1. Start
Created needed data including observation, truth, and initial analysis guess:
1. Modify `parameterControl.py` to set the global parameters
2. Generate needed data: `$ python dataRecorder.py` -> placed in `./initRecord`
3. OUTPUT:
  * `initRecord/observationOperator`: for all algorithm use
  * `initRecord/*ObservationNoiseType*/`:
    * observation in full/sparse
    * truth in full/sparse
    * initial error covariance
    * initial analysis state

### 2. Run Algorithm
Run DA algorithm to generate outcomes
1. Just choose one method to run `$ python *method*.py`
2. The data would be saved in "./*method*_record"
3. OUTPUT:
  * `*method*_record/`:
    * `*NoiseType*_*NoiseScale*_*variable*.txt`
