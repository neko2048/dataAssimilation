#!/bin/bash
echo "========== Generate Initial Data =========="
echo "Gaussian 0.2 fullOBSOPT" | python initValueGenerator.py 
# noiseType, noiseScale, gaussianRatio, observationOperatorType

echo "========== Run EKF =========="
python EKF.py

echo "========== Run 3DVar =========="
python 3DVar.py

echo "========== Run incremental 3DVar =========="
python incre3Var.py

echo "========== Run 4DVar =========="
python 4DVar.py

echo "========== Run incremental 4DVar =========="
python incre4DVar.py