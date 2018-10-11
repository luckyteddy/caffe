#include <algorithm>
#include <vector>
#include <iostream>

#include <boost/math/special_functions/next.hpp>

#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/context.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/platform.hpp"

#include "caffe/util/math_functions.hpp"

#include <clblast.h>      // NOLINT

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#include "caffe/caffe.hpp"

using namespace caffe;
using namespace std;


void to_gpu(float* cpu_p, cl_mem gpu_p, viennacl::ocl::context* ctx, int size){
  cl_int err;
  gpu_p = clCreateBuffer(ctx->handle().get(), CL_MEM_READ_WRITE, size, nullptr, &err);
  cout << "gpu_p = " << gpu_p << endl;
  //CHECK_EQ(0, err) << "OpenCL buffer allocation of size " << size << " failed.";
  //gpu_p = reinterpret_cast<void*>(cl_gpu_mem);
  if (err) cout << "buffer allocation fail" << endl;
  
  clEnqueueWriteBuffer(ctx->get_queue().handle().get(), gpu_p, CL_TRUE, 0, size, cpu_p, 0, NULL, NULL);
  ctx->get_queue().finish();
}

int main() {

  int M = 3;
  int N = 4;
  int K = 2;
  CBLAS_TRANSPOSE TransA = CblasNoTrans;
  CBLAS_TRANSPOSE TransB = CblasNoTrans; 
  float alpha = 1.0;
  float beta = 0.0;

  device* device_;
  int count = 0;
  int ctx_id;
  count = Caffe::EnumerateDevices(false);
  Caffe::SetDevices(std::vector<int>{0});
  Caffe::SetDevice(0);
  device_ = Caffe::GetDevice(0, true);
  viennacl::ocl::context& ctx = viennacl::ocl::get_context(device_->id());
  device_->FinishQueues();
 
  Blob<float> A(1,1,M,K);
  Blob<float> B(1,1,K,N);
  Blob<float> C(1,1,M,N);
  float*  cpu_pA = A.mutable_cpu_data();
  float*  cpu_pB = B.mutable_cpu_data();
  float*  cpu_pC = C.mutable_cpu_data();

  //float B[K*N];
  //float C[M*N];
  int id;

  cout << "A = " << endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      id = i * K + j;
      cpu_pA[id] = id;
      cout << cpu_pA[id] << " ";
    }
    cout << endl;
  }
  cout << "B = " << endl;
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      id = i * N + j;
      //B[id] = id;
      if (id % 2 == 0) {cpu_pB[id] = 1;} else {cpu_pB[id] = 2;}
      cout << cpu_pB[id] << " ";
    }
    cout << endl;
  }
  for (int i = 0; i < M*N; ++i) { cpu_pC[i] = 0;} 
  
  caffe_cpu_gemm<float>(TransA, TransB, M, N, K, alpha, cpu_pA, cpu_pB, beta, cpu_pC);

  cout << "CPU Result: C =" << endl;
  for (int i = 0; i < M; ++i) {
    for (int j =0 ; j < N; ++j) {
      cout << cpu_pC[i * N + j] << " ";
    }
    cout << endl;
  }

  float *gpu_pA = A.mutable_gpu_data();
  float *gpu_pB = B.mutable_gpu_data();
  float *gpu_pC = C.mutable_gpu_data();

  //void *gpu_pA = nullptr;
  //void *gpu_pB = nullptr;
  //void *gpu_pC = nullptr;

  to_gpu(cpu_pA, (cl_mem)gpu_pA, &ctx, M*K*sizeof(float));
  device_->FinishQueues();
  to_gpu(cpu_pB, (cl_mem)gpu_pB, &ctx, K*N*sizeof(float));
  to_gpu(cpu_pC, (cl_mem)gpu_pC, &ctx, M*N*sizeof(float));
  cout << "after to gpu" << gpu_pA << "; " << gpu_pB <<endl;
  greentea_gpu_gemm<float>(device_->id(), TransA, TransB, M, N, K, alpha, (cl_mem)gpu_pA, 0, (cl_mem)gpu_pB, 0, beta, (cl_mem)gpu_pC, 0);
//  cout << " GPU Result: C=" <<endl;
//  for (int i = 0; i < M; ++i) {
//    for (int j = 0; j< N ; ++j) {
//      cout << C[i * N + j] << " ";
//    }
//    cout << endl;
//  }

  caffe::Caffe::TeardownDevice(0);

}

