#include <iostream>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <cstdlib>

#include <time.h>
#include <sys/time.h>

using namespace std;

/***********************/
/* CUDA ERROR CHECKING */
/***********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/********/
/* MAIN */
/********/
int main() {

    srand(time(NULL));

    const int N = 10;

    float *h_vec = (float *)malloc(N * sizeof(float));
    for (int i=0; i<N; i++) {
        h_vec[i] = rand() / (float)(RAND_MAX);
        printf("h_vec[%i] = %f\n", i, h_vec[i]);
    }

    float *d_vec;
    gpuErrchk(cudaMalloc((void**)&d_vec, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_vec);

    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr + N);

    float max_value = max_ptr[0];
    printf("\nmaxinum value = %f\n", max_value);
    printf("Position = %i\n", &max_ptr[0] - &dev_ptr[0]);
    // for (int i = 0; i < 1; i++) {
    //   cout<< max_ptr[i] << endl;
    // }

}
// #include <thrust/device_vector.h>
// #include <thrust/extrema.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/reduce.h>
// #include <iostream>

// int main()
// {
//     const int arraySize = 10;

//     // Generate input data
//     thrust::device_vector<int> keys(arraySize);
//     thrust::device_vector<float> values(arraySize);
//     for (int i = 0; i < arraySize; ++i)
//     {
//         keys[i] = i % 3; // Assign keys to create groups
//         values[i] = static_cast<float>(i);
//     }

//     // Find the maximum value index within each group
//     thrust::counting_iterator<int> iter(0);
//     thrust::device_vector<int> groupIndices(arraySize);
//     thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), values.end(), thrust::equal_to<int>(), thrust::maximum<int>());

//     // Print the result
//     std::cout << "Group Indices: ";
//     for (int i = 0; i < groupIndices.size(); ++i)
//     {
//         std::cout << groupIndices[i] << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }

// #include <iostream>
// #include <thrust/copy.h>
// #include <thrust/reduce.h>
// #include <thrust/sort.h>
// #include <thrust/device_vector.h>
// #include <thrust/iterator/zip_iterator.h>
// #include <thrust/sequence.h>
// #include <thrust/functional.h>
// #include <cstdlib>

// #include <time.h>
// #include <sys/time.h>
// #define vsize 4

// // unsigned long long dtime_usec(unsigned long long start){

// //   timeval tv;
// //   gettimeofday(&tv, 0);
// //   return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
// // }

// // const size_t ksize = 4;
// // const size_t vsize = 4;
// // const int nTPB = 256;

// // struct my_max_func
// // {

// //   template <typename T1, typename T2>
// //   __host__ __device__
// //   T1 operator()(const T1 t1, const T2 t2){
// //     T1 res;
// //     if (thrust::get<0>(t1) > thrust::get<0>(t2)){
// //       thrust::get<0>(res) = thrust::get<0>(t1);
// //       thrust::get<1>(res) = thrust::get<1>(t1);}
// //     else {
// //       thrust::get<0>(res) = thrust::get<0>(t2);
// //       thrust::get<1>(res) = thrust::get<1>(t2);}
// //     return res;
// //     }
// // };

// // typedef union  {
// //   float floats[2];                 // floats[0] = maxvalue
// //   int ints[2];                     // ints[1] = maxindex
// //   unsigned long long int ulong;    // for atomic update
// // } my_atomics;


// // __device__ unsigned long long int my_atomicMax(unsigned long long int* address, float val1, int val2)
// // {
// //     my_atomics loc, loctest;
// //     loc.floats[0] = val1;
// //     loc.ints[1] = val2;
// //     loctest.ulong = *address;
// //     while (loctest.floats[0] <  val1)
// //       loctest.ulong = atomicCAS(address, loctest.ulong,  loc.ulong);
// //     return loctest.ulong;
// // }


// // __global__ void my_max_idx(const float *data, const int *keys,const int ds, my_atomics *res)
// // {

// //     int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
// //     if (idx < ds)
// //       my_atomicMax(&(res[keys[idx]].ulong), data[idx],idx);
// // }


// int main(){

//   float *vals;
//   int *keys;
//   my_atomics *results;
//   cudaMalloc(&keys, vsize*sizeof(int));
//   cudaMalloc(&vals, vsize*sizeof(float));
//   cudaMalloc(&results, ksize*sizeof(my_atomics));

//   cudaMemset(results, 0, ksize*sizeof(my_atomics)); // works because vals are all positive
//   cudaMemcpy(keys, h_keys, vsize*sizeof(int), cudaMemcpyHostToDevice);
//   cudaMemcpy(vals, h_vals, vsize*sizeof(float), cudaMemcpyHostToDevice);
//   std::cout << "CUDA time: " << et/(float)USECPSEC << "s" << std::endl;

// // verification

//   my_atomics *h_results = new my_atomics[ksize];
//   cudaMemcpy(h_results, results, ksize*sizeof(my_atomics), cudaMemcpyDeviceToHost);

//   my_max_idx<<<(vsize+nTPB-1)/nTPB, nTPB>>>(vals, keys, vsize, results);
//   cudaDeviceSynchronize();
//   // float h_vec[vsize];
//   // // thrust::generate(h_vec.begin(), h_vec.end(), rand);
//   // for (int i = 0; i < vsize; i++) {h_vec[i] = rand();}
//   // d_vals(h_vals, h_vals+vsize);
//   // thrust::device_vector<float> d_vec = h_vec;

//   // thrust::device_vector<float>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());

//   // unsigned int position = iter - d_vec.begin();
//   // float max_val = *iter;

//   // std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;
//   return 0;
// }