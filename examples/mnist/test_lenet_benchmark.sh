#!/usr/bin/env sh
set -e

./build/tools/caffe time --model=examples/mnist/lenet_train_test.prototxt $@ -gpu 0 

