#!/bin/bash
echo "========== Generate Initial Value =========="
python initValueGenerate.py

echo "========== run EKF =========="
python EKF.py

echo "========== run 3DVar =========="
python 3DVar.py

echo "========== run incre3DVar =========="
python incre3DVar.py

echo "========== run 4DVar =========="
python 4DVar.py

echo "========== run incre4DVar =========="
python incre4DVar.py

echo "========== run NoDA =========="
python noDA.py