#!/usr/bin/env bash

set -e
set -o pipefail

python amsgrad_mnist.py --learning_rate 0.5
python amsgrad_mnist.py --learning_rate 0.05
python amsgrad_mnist.py --learning_rate 0.005
python amsgrad_mnist.py --learning_rate 0.0005
python amsgrad_mnist.py --learning_rate 0.00005

python sgd_mnist.py --learning_rate 0.5
python sgd_mnist.py --learning_rate 0.05
python sgd_mnist.py --learning_rate 0.005
python sgd_mnist.py --learning_rate 0.0005
python sgd_mnist.py --learning_rate 0.00005
