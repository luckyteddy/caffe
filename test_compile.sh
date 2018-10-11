g++ -std=c++11 test.cpp -DUSE_GREENTEA -DUSE_CLBLAST -I/home/sdp/xu/caffe/include -I./build/src -L/home/sdp/xu/caffe/build/lib -lcaffe -lclblast -lboost_system -lOpenCL -g3 -o test.out
