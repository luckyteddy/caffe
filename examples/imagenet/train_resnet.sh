nohup ./build/tools/caffe train --solver=models/ResNet-50/solver.prototxt $@ -gpu 0 >& Res50.log < /dev/null&
