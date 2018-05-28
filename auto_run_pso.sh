#!/usr/bin/env bash

set -e
set -o pipefail

python pso_fcn_mnist.py --hybrid --vr --lr 0.5
python pso_fcn_mnist.py --hybrid --vr --lr 0.05
python pso_fcn_mnist.py --hybrid --vr --lr 0.005
python pso_fcn_mnist.py --hybrid --vr --lr 0.0005
python pso_fcn_mnist.py --hybrid --vr --lr 0.00005

python pso_fcn_mnist.py --hybrid --vr --mvdec 1
python pso_fcn_mnist.py --hybrid --vr --mvdec 0.8
python pso_fcn_mnist.py --hybrid --vr --mvdec 0.6
python pso_fcn_mnist.py --hybrid --vr --mvdec 0.4
python pso_fcn_mnist.py --hybrid --vr --mvdec 0.2

