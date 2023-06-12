#include <iostream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <sys/time.h>
#include <vector>
#include <set>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <functional>
// #include <helper_cuda.h>








using namespace std;








//Define the parameters if not defined externally
#ifndef cmd_def
#define InputNum 4  // Number of input data points (instances)
#define NUM_FEATURE 4  // Number of features in an instance
#define MaxDepth 10  // Number of features in an instance
# define MinimumSplitNumInstances 1
# define MaxNodeNum (static_cast<int>(pow(2, MaxDepth)) - 1)
#endif
#define VTYPE float
# define DataSize (NUM_FEATURE * InputNum)
# define Lambda 1
// # define numRows 8
// # define numCols 4
# define NUM_THREAD 512
# define Gamma 0
# define Left true
# define Right false
# define KB (1 << 10)
# define MB (1 << 20)
# define GB (1 << 30)
# define DynamicMemorySize (1 * GB)


class node
{
public:

    VTYPE predicted_value;
    int num_instances; // number of instances
    int left_child_id;
    int right_child_id;
    int start_index; // start index in data
    int feature_id; // the feature we use to split the node
    int split_index;  // the feature we use to split the node
    VTYPE feature_threshold; // the threshold of the feature value; if it is larger than threshold, it goes to the right child, otherwise the left child

    // __host__ __device__ node()
    // {
    //     predicted_value = -1;
    //     node_id = -1;
    //     num_instances = -1; // number of instances
    //     level = -1;
    //     split_index = -1;
    //     left_child_id = -1;
    //     right_child_id = -1;
    //     training_loss = 0.0;
    //     is_leaf = false;
    //     start_index = 0;
    //     feature_id = -1;
    //     feature_threshold = 0.0;
    //     index_in_segment = -1;
    // }
};



#include <utility>


class attribute_id_pair {
public:
   VTYPE attribute_value;
   int instance_id;
};




// read the input file (e.g. ./benchmarks/CASP) return global memory array
// sort the input with attribute values and return
// create y values
// void read_input(string input_path, vector<attribute_id_pair>& data) {




// }
void read_input(attribute_id_pair* data, VTYPE* label) {
// void read_input(string input_path, std::vector<attribute_id_pair>& data, std::vector<VTYPE>& label) {
/*
// include libxl.h for this run//
    Book* book = xlCreateBook();
//  Book*  book = xlCreateBook();
    if (!book) {
        std::cout << "Error creating book." << std::endl;
        return;
    }


    if (book->load(input_path.c_str())) {
        libxl::Sheet* sheet = book->getSheet(0); // Assuming the first sheet
        if (sheet) {
            const int numRows = sheet->lastRow() ;
            const int numCols = sheet->lastCol() + 1;

            data.reserve(numRows*numCols);
            label.reserve(numRows);

        for (int i = 0; i < numCols*numRows; ++i) {
            for (int col = 0; col < numCols; ++col) {
                for (int row = 1; row < numRows; ++row) {
                    attrIdPair.instance_id = i;
                    attribute_id_pair attrIdPair;

                    attrIdPair.attribute_value = (sheet->readNum(row, col));
                    data.push_back(attrIdPair);
                    label.push_back(sheet->readNum(row, 1));
                    }
                }

            }
        }
    }

    else {
        std::cout << "Error loading the workbook." << std::endl;
    }

    book->release();
*/
    // data.reserve(numRows*numCols);
    // label.reserve(numRows);

    // for (int i = 0; i < numCols*numRows; ++i) {
    //     for (int col = 0; col < numCols; ++col) {
    //         for (int row = 1; row < numRows; ++row) {
    //             attribute_id_pair attrIdPair;
    //             attrIdPair.instance_id = i;
    //             attrIdPair.attribute_value = 1;
    //             int label_num;
    //             label_num = i;
        
    //             data.push_back(attrIdPair);
    //             label.push_back(label_num);
    //             }
    //         }  
    //     }
    // for (int i = 0; i < NUM_FEATURE; i++) {
    //     for (int j = 0; j < InputNum; j++) {
    //         attribute_id_pair pair;
    //         int id = i * InputNum + j;
    //         pair.attribute_value = VTYPE(id);
    //         pair.instance_id = id;
    //         data[id] = pair;
    //     }
    // }

    for (int i = 0; i < DataSize; i++){
        label[i] = i;
    }
    data[0].attribute_value = 0.160000;
    data[0].instance_id = 0;
    data[1].attribute_value = 1.580000;
    data[1].instance_id = 3;
    data[2].attribute_value = 2.830000;
    data[2].instance_id = 2;
    data[3].attribute_value = 3.760000;
    data[3].instance_id = 1;
    data[4].attribute_value = 0.160000;
    data[4].instance_id = 0;
    data[5].attribute_value = 1.580000;
    data[5].instance_id = 1;
    data[6].attribute_value = 2.830000;
    data[6].instance_id = 2;
    data[7].attribute_value = 3.760000;
    data[7].instance_id = 3;
    data[8].attribute_value = 0.160000;
    data[8].instance_id = 0;
    data[9].attribute_value = 1.580000;
    data[9].instance_id = 3;
    data[10].attribute_value = 2.830000;
    data[10].instance_id = 2;
    data[11].attribute_value = 3.760000;
    data[11].instance_id = 1;
    data[12].attribute_value = 0.160000;
    data[12].instance_id = 0;
    data[13].attribute_value = 1.580000;
    data[13].instance_id = 1;
    data[14].attribute_value = 2.830000;
    data[14].instance_id = 2;
    data[15].attribute_value = 3.760000;
    data[15].instance_id = 3;

    label[0] = 0;
    label[1] = 1;
    label[2] = 2;
    label[3] = 3;
}








inline void cuda_check_error() {
    auto err = cudaGetLastError();
    if(err) {
      std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
      exit(0);
    }
  }




void fill_data(attribute_id_pair* data) {
   for (int i = 0; i < DataSize; i++) {
       attribute_id_pair pair;
       pair.attribute_value = static_cast<VTYPE>(rand() % 101) / 100.0; // Generate a random float between 0 and 1
       pair.instance_id = rand() % 101; // Generate a random integer between 0 and 100
       data[i] = pair;
   }
}
















void fill_label(VTYPE* label){
 int i;
 for (i = 0; i < DataSize; i++){
   label[i] = i;
 }
}








static uint64_t usec;








static __inline__ uint64_t gettime(void) {
 struct timeval tv;
 gettimeofday(&tv, NULL);
 return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec));
}








__attribute__ ((noinline))  void begin_roi() {
 usec=gettime();
}
__attribute__ ((noinline))  void end_roi()   {
 usec=(gettime()-usec);
 std::cout << "elapsed (sec): " << usec/1000000.0 << "\n";
}

__global__ void get_gradient(node* d_nodes, attribute_id_pair* d_data, VTYPE* d_label, VTYPE* d_buffer, int node_start_id) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int start_index = cur_node.start_index;
    int start_index_of_this_thread = cur_node.start_index + thread_idx;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    int instance_id;
    VTYPE y, y_hat, gradient;

    for (int index = start_index_of_this_thread; index < end_index; index += blockDim.x) {
        attribute_id_pair pair = d_data[index];
        instance_id = pair.instance_id;
        y = d_label[instance_id];
        y_hat = cur_node.predicted_value;
        gradient = 2 * (y_hat - y);
        d_buffer[index] = gradient;
    }
}

// a segment is a feature in a node. A feature contains some instances
__global__ void set_key_segment(node* d_nodes, int* d_key, int node_start_id) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int start_index = cur_node.start_index;
    int start_index_of_this_thread = cur_node.start_index + thread_idx;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    int key, feature_num;
    

    for (int index = start_index_of_this_thread; index < end_index; index += blockDim.x) {
        feature_num = (index / num_instances);
        key = node_id * NUM_FEATURE + feature_num;
        d_key[index] = key;
    }
}


__global__ void get_gain(node* d_nodes, VTYPE* d_buffer, int node_start_id) { // d_buffer is prefix sum so far
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    

    int start_index_of_this_thread = cur_node.start_index + thread_idx;
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;

    // // debug
    // __syncthreads();
    // if (thread_idx == 0){
    //     printf("G\n");
    //     printf("blockIdx.x  %d node_id %d\n", blockIdx.x, node_id);
    //     for (int index = start_index_of_this_thread; index < end_index; index ++) {
    //         printf("d_buffer[%d] %f\n", index, d_buffer[index]);
    //     }
    // }
    // __syncthreads();
    // // end of debug

    VTYPE gain, H_l, H_r, G_l, G_r, G_L_plus_G_r;
    int index_in_segment, last_local_data_index;
    int num_left_instance, num_right_instance;

    // skip the last instance
    for (int index = start_index_of_this_thread; index < end_index; index += blockDim.x) {
        index_in_segment = index % num_instances;
        if (index_in_segment == NUM_FEATURE - 1) { // the last instance in the feature, cannot split because the right child has no instances
            //d_buffer[index] = 0;
            continue;
        }
        num_left_instance = index_in_segment + 1;
        num_right_instance = num_instances - num_left_instance;
        if (num_left_instance < MinimumSplitNumInstances || num_right_instance < MinimumSplitNumInstances) {
            d_buffer[index] = 0;
            continue;
        }
        last_local_data_index = (((index / num_instances) + 1) * num_instances) - 1;
        G_l = d_buffer[index];
        G_r = d_buffer[last_local_data_index] - G_l;
        H_l = (index_in_segment + 1) * 2;
        H_r = (num_instances - index_in_segment - 1) * 2;
        G_L_plus_G_r = G_l + G_r;
        gain = ((G_l * G_l / (H_l + Lambda)) + (G_r * G_r / (H_r + Lambda)) - (G_L_plus_G_r * G_L_plus_G_r / (H_l + H_r - Lambda))) * 0.5;
        d_buffer[index] = gain;

        // // debug
        // if (thread_idx == 0){
        //     printf("G_l %f G_r %f H_l %f H_r %f gain %f index %d\n", G_l, G_r, H_l, H_r, gain, index);
        // }
        // // end of debug
    }

    __syncthreads();

    // write 0 into last instance
    int loop_start_index = (cur_node.start_index + (num_instances * thread_idx) - 1);
    int increment = num_instances * blockDim.x;
    for (int index = loop_start_index; index < end_index; index += increment) {
        d_buffer[index] = 0;
    }

    // // debug
    // __syncthreads();
    // if (thread_idx == 0){
    //     printf("gain\n");
    //     printf("blockIdx.x  %d node_id %d\n", blockIdx.x, node_id);
    //     for (int index = start_index_of_this_thread; index < end_index; index ++) {
    //         printf("d_buffer[%d] %f\n", index, d_buffer[index]);
    //     }
    // }
    // __syncthreads();
    // // end of debug
}

__global__ void get_best_split_point(node* d_nodes, VTYPE* d_buffer, int node_start_id, attribute_id_pair* d_data) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int start_index = cur_node.start_index;
    int start_index_of_this_thread = start_index + thread_idx;
    int num_instances = cur_node.num_instances;
    int node_size = num_instances * NUM_FEATURE;
    if (thread_idx >= node_size) {
        return;
    }
    int num_thread_per_block = min(blockDim.x, node_size);
    int end_index = start_index + node_size;
    VTYPE max_value = 0;
    VTYPE compared_value;
    int max_index = -1;
    for (int index = start_index_of_this_thread; index < end_index; index += num_thread_per_block) {
        compared_value = d_buffer[index];
        if (compared_value > max_value) {
            max_value = compared_value;
            max_index = index;
        }
        // printf("max_value %f, max_index %d thread_idx %d\n", max_value, max_index, thread_idx);
    }
    extern __shared__ VTYPE shared_mem[];
    VTYPE* max_values = reinterpret_cast<VTYPE*>(shared_mem);
    int* max_indices = reinterpret_cast<int*>(shared_mem + num_thread_per_block * sizeof(VTYPE));

    int last_num_active_thread = num_thread_per_block;
    int num_active_thread = (last_num_active_thread + 1) / 2;
    int compared_thread_idx, compared_index;
    while (last_num_active_thread > 1) {
        if (thread_idx >= num_active_thread && thread_idx < last_num_active_thread) {
            max_values[thread_idx] = max_value;
            max_indices[thread_idx] = max_index;
            // if (thread_idx == 5){
            //     printf("max_value %f, max_index %d thread_idx %d max_values[%d] %f max_indices[%d] %f\n", 
            //     max_value, max_index, thread_idx, thread_idx, max_values[thread_idx], thread_idx, max_indices[thread_idx]);
            // }
        }
        __syncthreads();
        compared_thread_idx = thread_idx + num_active_thread;
        compared_index = cur_node.start_index + compared_thread_idx;
        if (thread_idx < num_active_thread && compared_thread_idx < last_num_active_thread) {
            compared_value = max_values[compared_index];
            if (compared_value > max_value) {
                max_value = compared_value;
                max_index = max_indices[compared_index];
            }
        }
        last_num_active_thread = num_active_thread;
        num_active_thread = (num_active_thread + 1) / 2;
    }

    if (thread_idx == 0) {
        if (max_value > Gamma) {
            d_nodes[node_id].split_index = max_index;
            int feature_id = (max_index - start_index) / num_instances;
            d_nodes[node_id].feature_id = feature_id;
            VTYPE feature_threshold = (d_data[max_index].attribute_value + d_data[max_index+1].attribute_value) / 2;
            d_nodes[node_id].feature_threshold = feature_threshold;
        }
        // // debug
        // printf("max_value %f, max_index %d thread_idx %d feature_id %d node_id %d\n", 
        // max_value, max_index, thread_idx, d_nodes[node_id].feature_id, node_id);
    }
    
    // printf("1 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    // // debug
    // __syncthreads();
    // if (thread_idx == 0){
    //     printf("max gain\n");
    //     printf("max_value %f, max_index%d\n", max_value, max_index);
    // }
    // __syncthreads();
    // // end of debug
}
__global__ void set_counter(node* d_nodes, VTYPE* d_buffer, int* d_counter, int * d_key, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data) {
    // the range of counter is [2*blockDim.x:4*BlocDim.x]
    
    // printf("2 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int split_index = cur_node.split_index;
    if (split_index == -1) { // the entire node does not need to split
        return;
    }
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int node_size = num_instances * NUM_FEATURE;
    if (thread_idx >= node_size) {
        return;
    }
    // printf("2.1 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    // int num_thread_per_block = blockDim.x;
    int num_thread_per_block = min(blockDim.x, node_size);
    int global_counter_start_index = min(start_index, 2 * blockIdx.x * blockDim.x);
    int left_counter_index = global_counter_start_index + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    //printf("2.2 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    // reset key and counter
    for (int index = left_counter_index; index <= right_counter_index; index += num_thread_per_block) { // can move into if
        d_key[index] = node_id;
        d_counter[index] = 0; // may not need this
        //printf("index %d global_counter_start_index %d start_index %d left_counter_index %d right_counter_index %d\n", index, global_counter_start_index,start_index,left_counter_index,right_counter_index);
    }
    
    // printf("2.4 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    int local_counter[2] = {0, 0};

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    int feature_id = cur_node.feature_id;
    int split_start_index = start_index + (feature_id * num_instances);
    int end_index = start_index_of_this_thread + num_instances_per_thread;
    int instance_id;
    int left_instance_id;
    bool go_left = false;
    
    // printf("2.5 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    //printf("split_start_index %d feature_id %d split_index %d node_id %d\n",split_start_index, feature_id,split_index,node_id);
    for (int index = start_index_of_this_thread; index < end_index; index++) {
        instance_id = d_data[index].instance_id;
        go_left = false;
        for (int left_index = split_start_index; left_index <= split_index; left_index++) {
            left_instance_id = d_data[left_index].instance_id;
            if (instance_id == left_instance_id) {
                go_left = true;
                break;
            }
        }
        if (go_left) {
            d_split_direction[index] = Left;
            local_counter[0] ++;
        } else {
            d_split_direction[index] = Right;
            local_counter[1] ++;
        }
    }
    d_counter[left_counter_index] = local_counter[0];
    d_counter[right_counter_index] = local_counter[1];
    
    // printf("3 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    // debug
    // printf("d_counter[%d] %d thread_idx %d node_id %d local_counter[0] %d left\n", left_counter_index, d_counter[left_counter_index], thread_idx, node_id, local_counter[0]);
    // printf("d_counter[%d] %d thread_idx %d node_id %d local_counter[1] %d right\n", right_counter_index, d_counter[right_counter_index], thread_idx, node_id, local_counter[1]);
    // if (thread_idx == 0) {

    //     for (int i = 2 * blockIdx.x * blockDim.x; i < min(2 * (blockIdx.x + 1) * blockDim.x, DataSize); i++) {
    //         printf("d_key[%d] %d\n", i, d_key[i]);
    //     }
    // }
}
__global__ void creat_child_nodes(node* d_nodes, int node_start_id, int* d_lock, int* d_num_node_next_level, int num_node_cur_level) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int split_index = cur_node.split_index;
    if (split_index == -1) {
        return;
    }
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int node_size = num_instances * NUM_FEATURE;
    if (thread_idx >= node_size) {
        return;
    }
    // get child node id
    __shared__ int num_node_next_level;
    if (thread_idx == 0) {
        bool wait_lock = true;
        while (wait_lock) {
            if(atomicCAS(d_lock, 0, 1) == 0){
                num_node_next_level = *d_num_node_next_level;
                atomicAdd(d_num_node_next_level, 2);
                atomicExch(d_lock, 0);
                wait_lock = false;
            }
        }
    }
    if (thread_idx == 0) {
        int next_node_start_id = node_start_id + num_node_cur_level;
        int left_child_id = next_node_start_id + num_node_next_level;
        int right_child_id = left_child_id + 1;
        cur_node.left_child_id = left_child_id;
        cur_node.right_child_id = right_child_id;
        d_nodes[node_id] = cur_node;

        node left_child;
        int left_child_num_instance = ((split_index - start_index) % num_instances) + 1;
        left_child.num_instances = left_child_num_instance;
        left_child.start_index = start_index;
        left_child.predicted_value = -1;
        left_child.left_child_id = -1;
        left_child.right_child_id = -1;
        left_child.feature_id = -1;
        left_child.split_index = -1;
        left_child.feature_threshold = -1;
        d_nodes[left_child_id] = left_child;

        node right_child;
        right_child.num_instances = num_instances - left_child_num_instance;
        right_child.start_index = start_index + (left_child_num_instance * NUM_FEATURE);
        right_child.predicted_value = -1;
        right_child.left_child_id = -1;
        right_child.right_child_id = -1;
        right_child.feature_id = -1;
        right_child.split_index = -1;
        right_child.feature_threshold = -1;
        d_nodes[right_child_id] = right_child;

        // VTYPE predicted_value;
        // int num_instances; // number of instances
        // int left_child_id;
        // int right_child_id;
        // int start_index; // start index in data
        // int feature_id; // the feature we use to split the node
        // int split_index;  // the feature we use to split the node
        // VTYPE feature_threshold; // 
    }
}
__global__ void split_node(node* d_nodes, int* d_counter, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data, attribute_id_pair* d_new_data) {
    // printf("4 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int split_index = cur_node.split_index;
    if (split_index == -1) {
        return;
    }
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int node_size = num_instances * NUM_FEATURE;
    if (thread_idx >= node_size) {
        return;
    }
    int num_thread_per_block = min(blockDim.x, node_size);
    int global_counter_start_index = min(start_index, 2 * blockIdx.x * blockDim.x);
    int left_counter_index = global_counter_start_index + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    int local_counter[2];
    local_counter[0] = d_counter[left_counter_index];
    local_counter[1] = d_counter[right_counter_index];

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    // int feature_id = d_nodes[node_id].feature_id;
    // int split_start_index = start_index + (feature_id * num_instances);
    int end_index = start_index_of_this_thread + num_instances_per_thread;
    // int instance_id;
    // int left_instance_id;
    bool direction;
    int new_index;
    attribute_id_pair cur_data;
    // printf("start_index_of_this_thread %d, end_index %d thread_idx %d\n", start_index_of_this_thread, end_index,thread_idx);
    for (int index = start_index_of_this_thread; index < end_index; index++) {
        direction = d_split_direction[index];
        if (direction == Left) {
            new_index = start_index + local_counter[0];
            local_counter[0] ++;
        } else {
            new_index = start_index + local_counter[1];
            local_counter[1] ++;
        }
        cur_data = d_data[index];
        d_new_data[new_index] = cur_data;
        // printf("");
        // printf("new_index %d\n", new_index);
        // // debug
        // printf("index %d, direction %d, start_index %d, new_index %d, thread_idx %d\n", index, direction, start_index, new_index, thread_idx);
        // if (cur_data.instance_id == 3) {
        //     printf("d_new_data[%d].instance_id %d, d_new_data[%d].attribute_value %f index %d\n",
        //      new_index, d_new_data[new_index].instance_id, new_index, d_new_data[new_index].attribute_value, index);
        // }
    }

    // printf("hello\n");
    // create two new nodes
    // printf("create two new nodes\n");


    
    // printf("2 d_nodes[0].split_index %d\n",d_nodes[0].split_index);
}

__global__ void set_key_buffer_for_prediction_value(node* d_nodes, VTYPE* d_buffer, int node_start_id, attribute_id_pair* d_data, int num_node_cur_level, int* d_key, VTYPE* d_label) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    __syncthreads();
    int num_instances = cur_node.num_instances;
    int start_index = cur_node.start_index;
    // int start_index_of_this_thread = cur_node.start_index + thread_idx;
    // int end_index = start_index + num_instances;
    int start_local_data_index = thread_idx;
    int start_buffer_index = start_index / NUM_FEATURE;
    int buffer_index;
    int data_index;
    int instance_id;
    // if (thread_idx < InputNum){
    //     printf("thread_idx %d start_local_data_index %d (start_local_data_index < num_instances) %d\n", thread_idx, start_local_data_index, (start_local_data_index < num_instances));
    // }
    for (int local_data_index = start_local_data_index; local_data_index < num_instances; local_data_index += blockDim.x) {
        buffer_index = start_buffer_index + local_data_index;
        data_index = start_index + local_data_index;
        instance_id = d_data[data_index].instance_id;
        d_buffer[buffer_index] = d_label[instance_id];
        d_key[buffer_index] = node_id;
        // printf("data_index %d, instance_id %d, buffer_index %d d_buffer[buffer_index] %f d_label[instance_id] %f num_instances %d local_data_index %d thread_idx %d\n", data_index, instance_id, buffer_index, 
        // d_buffer[buffer_index], d_label[instance_id], num_instances, local_data_index, thread_idx);
    }

    // // debug
    // __syncthreads();
    // if (thread_idx == 0) {
    //     // printf("before sum\nbuffer\n");
    //     // for (int i = 0; i < InputNum; i++){
    //     //     printf("d_label[%d] %f\n", i, d_label[i]);
    //     // }
    //     // for (int i = 0; i < InputNum; i++){
    //     //     printf("d_label[%d] %f\n", i, d_label[i]);
    //     // }
    //     // for (int i = 0; i < InputNum; i++){
    //     //     printf("d_buffer[%d] %f\n", i, d_buffer[i]);
    //     // }
    //     // printf("key\n");
    //     // for (int i = 0; i < InputNum; i++){
    //     //     printf("d_key[%d] %f\n", i, d_key[i]);
    //     // }
    // }
}

__global__ void set_prediction_value(node* d_nodes, VTYPE* d_buffer, int node_start_id, int num_node_cur_level) {
    // printf("hello\n");
    int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("global_thread_idx %d\n", global_thread_idx);
    // if(global_thread_idx ==0) {
    //     printf("sum %f\n" , d_buffer[0]);
    // }
    if (global_thread_idx >= num_node_cur_level) {
        return;
    }
    int node_id = node_start_id + global_thread_idx;
    node cur_node = d_nodes[node_id];
    int num_instances = cur_node.num_instances;
    VTYPE predicted_value = d_buffer[global_thread_idx] / num_instances;
    cur_node.predicted_value = predicted_value;
    d_nodes[node_id] = cur_node;
}
int main(void) {
    node nodes[MaxNodeNum];
    node root;
    root.num_instances = InputNum;
    root.start_index = 0;
    // ... set other member variables as needed
    nodes[0] = root;
    // Define the aligned array
    
    
    attribute_id_pair data[DataSize];
    VTYPE label [InputNum];
    read_input(data, label);
    cout << "starting program\n";
    // fill_data(data);
    // fill_label(label);


    attribute_id_pair *d_data;
    // checkCudaErrors(cudaMalloc((void **)(&d_data), sizeof(attribute_id_pair) * DataSize));
    cudaMalloc((void **)(&d_data), sizeof(attribute_id_pair) * DataSize);
    // checkCudaErrors(cudaMemcpy(d_data, data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyHostToDevice));
    cudaMemcpy(d_data, data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyHostToDevice);
    attribute_id_pair *d_new_data;
    // checkCudaErrors(cudaMalloc((void **)(&d_new_data), sizeof(attribute_id_pair) * DataSize));
    cudaMalloc((void **)(&d_new_data), sizeof(attribute_id_pair) * DataSize);

    int *d_key;
    // checkCudaErrors(cudaMalloc((void **)(&d_key), sizeof(int) * DataSize));
    cudaMalloc((void **)(&d_key), sizeof(int) * DataSize);

    VTYPE *d_buffer;
    // checkCudaErrors(cudaMalloc((void **)(&d_buffer), sizeof(VTYPE) * DataSize));
    cudaMalloc((void **)(&d_buffer), sizeof(VTYPE) * DataSize);

    
    // int *d_best_split_index;
    // cudaMalloc((void **)(&d_best_split_index), sizeof(int) * DataSize);

    
    bool *d_split_direction;
    // checkCudaErrors(cudaMalloc((void **)(&d_split_direction), sizeof(bool) * DataSize));
    cudaMalloc((void **)(&d_split_direction), sizeof(bool) * DataSize);

    VTYPE *d_label;
    // checkCudaErrors(cudaMalloc((void **)(&d_label), sizeof(VTYPE) * DataSize)); // Use d_label instead of label
    cudaMalloc((void **)(&d_label), sizeof(VTYPE) * InputNum);
    // checkCudaErrors(cudaMemcpy(d_label, label, sizeof(VTYPE) * DataSize, cudaMemcpyHostToDevice));
    cudaMemcpy(d_label, label, sizeof(VTYPE) * InputNum, cudaMemcpyHostToDevice);

    // Allocate memory on the GPU
    node *d_nodes;
    // checkCudaErrors(cudaMalloc((void**)&d_nodes, sizeof(node) * MaxNodeNum));
    cudaMalloc((void**)&d_nodes, sizeof(node) * MaxNodeNum);
    // Copy the nodes array to the GPU
    // checkCudaErrors(cudaMemcpy(d_nodes, nodes, sizeof(node) * MaxNodeNum, cudaMemcpyHostToDevice));
    cudaMemcpy(d_nodes, nodes, sizeof(node) * MaxNodeNum, cudaMemcpyHostToDevice);
    


 

    int level = 0;

    int *d_num_node_next_level;
    // checkCudaErrors(cudaMalloc((int **)(&d_num_node_next_level), sizeof(int))); // Use d_label instead of label
    cudaMalloc((int **)(&d_num_node_next_level), sizeof(int));

    // int *d_total_num_nodes;
    // cudaMalloc((int **)(&d_total_num_nodes), sizeof(int)); // Use d_label instead of label

    // const int zero = 0;
    int* d_lock;
    // checkCudaErrors(cudaMalloc((int **)(&d_lock), sizeof(int))); // Use d_label instead of label
    cudaMalloc((int **)(&d_lock), sizeof(int));
    // checkCudaErrors(cudaMemset(d_lock, 0, sizeof(int)));
    cudaMemset(d_lock, 0, sizeof(int));
    cuda_check_error();

    

    int num_node_cur_level = 1;
    int node_start_id = 0;
    dim3 grid_size(num_node_cur_level, 0, 0);
    dim3 block_size(NUM_THREAD, 0, 0);
    int num_cache_entry_per_block;
    int dynamic_memory_size;
    while (level < MaxDepth - 1 && (num_node_cur_level > 0)) {
        // printf("level %d\n", level);
        // checkCudaErrors(cudaMemset(d_num_node_next_level, 0, sizeof(int)));
        cudaMemset(d_num_node_next_level, 0, sizeof(int));
        // cudaMemcpy(d_total_num_nodes, &total_num_nodes, sizeof(int), cudaMemcpyHostToDevice);
        grid_size = dim3(num_node_cur_level);
        block_size = dim3(NUM_THREAD);
        // printf("set_key_buffer_for_prediction_value\n");
        set_key_buffer_for_prediction_value<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id, d_data, num_node_cur_level, d_key, d_label);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("thrust::reduce_by_key\n");
        auto new_end = thrust::reduce_by_key(thrust::device, d_key, d_key + InputNum, d_buffer, d_key, d_buffer);
        cudaDeviceSynchronize();
        cuda_check_error();
        // auto new_end = thrust::reduce_by_key(thrust::device, d_key, d_key + 1, d_buffer, d_key, d_buffer);
        grid_size = dim3((num_node_cur_level + block_size.x - 1) / block_size.x);
        // printf("set_prediction_value grid_size %d block_size %d\n", grid_size.x, block_size.x);
        set_prediction_value<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id, num_node_cur_level);
        cudaDeviceSynchronize();
        cuda_check_error();
        grid_size = dim3(num_node_cur_level);
        // printf("get_gradient\n");
        get_gradient<<<grid_size, block_size>>>(d_nodes, d_data, d_label, d_buffer, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("set_key_segmen\n");
        set_key_segment<<<grid_size, block_size>>>(d_nodes, d_key, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        thrust::inclusive_scan_by_key(thrust::device, d_key, d_key + DataSize, d_buffer, d_buffer); // find G
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("get_gain\n");
        get_gain<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        // it is inclusive so for split point at index i, data i belong to the left child
        num_cache_entry_per_block = DynamicMemorySize / (sizeof(VTYPE) + sizeof(int)) / (num_node_cur_level);
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        dynamic_memory_size = block_size.x * (sizeof(VTYPE) + sizeof(int));
        // printf("dynamic_memory_size %d\n", dynamic_memory_size);
        // printf("get_best_split_point\n");
        get_best_split_point<<<grid_size, block_size, dynamic_memory_size>>>(d_nodes, d_buffer, node_start_id, d_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        num_cache_entry_per_block = DynamicMemorySize / sizeof(int) / (num_node_cur_level) / 2;
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        // printf("block size %d num_cache_entry_per_block %d NUM_THREAD %d\n", block_size.x, num_cache_entry_per_block, NUM_THREAD);
        int *d_counter;
        // checkCudaErrors(cudaMalloc((void **)(&d_counter), sizeof(int) * block_size.x * (num_node_cur_level) * 2));
        cudaMalloc((void **)(&d_counter), sizeof(int) * block_size.x * (num_node_cur_level) * 2);
        cudaDeviceSynchronize();
        cuda_check_error();
        // debug
        // cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
        // printf("nodes[0].feature_id %d cpu\n", nodes[0].feature_id);
        // end debug
        
        cudaMemset(d_key, -1, sizeof(int) * DataSize);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("set_counter\n");
        set_counter<<<grid_size, block_size>>>(d_nodes, d_buffer, d_counter, d_key, node_start_id, d_split_direction, d_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        thrust::exclusive_scan_by_key(thrust::device, d_key, d_key + (block_size.x * 2), d_counter, d_counter); // find G
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("creat_child_nodes\n");
        creat_child_nodes<<<grid_size, 1>>>(d_nodes, node_start_id, d_lock, d_num_node_next_level, num_node_cur_level);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("split_node\n");
        split_node<<<grid_size, block_size>>>(d_nodes, d_counter, node_start_id, d_split_direction, d_data, d_new_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        
        // // debug
        // printf("before copy\n");
        // attribute_id_pair new_data[DataSize];
        // cudaMemcpy(data, d_data, sizeof(attribute_id_pair)* DataSize, cudaMemcpyDeviceToHost);
        // cudaMemcpy(new_data, d_new_data, sizeof(attribute_id_pair)* DataSize, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < DataSize; i ++) {
        //     printf("data[%d].instance_id %d, data[%d].attribute_value %f\n", i, data[i].instance_id, i, data[i].attribute_value);
        // }
        // for (int i = 0; i < DataSize; i ++) {
        //     printf("new_data[%d].instance_id %d, new_data[%d].attribute_value %f\n", i, new_data[i].instance_id, i, new_data[i].attribute_value);
        // }
        // // end debug

        cudaMemcpy(d_data, d_new_data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("cudaFree\n");
        cudaFree(d_counter);
        // printf("cudaDeviceSynchronize\n");
        cudaDeviceSynchronize();
        cuda_check_error();
        // debug
        // printf("after copy\n");
        // cudaMemcpy(data, d_data, sizeof(attribute_id_pair)* DataSize, cudaMemcpyDeviceToHost);
        // for (int i = 0; i < DataSize; i ++) {
        //     printf("data[%d].instance_id %d, data[%d].attribute_value %f\n", i, data[i].instance_id, i, data[i].attribute_value);
        // }
        // end debug
        // cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
        // printf("cudaMemcpy\n");
        // checkCudaErrors(cudaMemcpy(&num_node_cur_level, d_num_node_next_level, sizeof(int), cudaMemcpyDeviceToHost));
        node_start_id += num_node_cur_level;
        level++;
        cudaMemcpy(&num_node_cur_level, d_num_node_next_level, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cuda_check_error();
        // printf("num_node_cur_level %d level %d\n",num_node_cur_level, level);
        // cudaMemcpy(&total_num_nodes, node_start_id, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("node_start_id += num_node_cur_level\n");
    }
    if (num_node_cur_level > 0) {
        grid_size.x = num_node_cur_level;
        block_size.x = NUM_THREAD;
        // printf("out of loop\nnum_node_cur_level %d, node_start_id %d\n",num_node_cur_level,node_start_id);
        // printf("set_key_buffer_for_prediction_value\n");
        set_key_buffer_for_prediction_value<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id, d_data, num_node_cur_level, d_key, d_label);
        cudaDeviceSynchronize();
        cuda_check_error();
        auto new_end = thrust::reduce_by_key(thrust::device, d_key, d_key + InputNum, d_buffer, d_key, d_buffer);
        cudaDeviceSynchronize();
        cuda_check_error();
        grid_size.x = (num_node_cur_level + block_size.x - 1) / block_size.x;
        // printf("set_prediction_value\n");
        set_prediction_value<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id, num_node_cur_level);
        cudaDeviceSynchronize();
        cuda_check_error();
    }
    cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cuda_check_error();
    // begin_roi();
    // classifier_GPU<<<grid_size, grid_size>>>(d_data);
    // cudaDeviceSynchronize();
    // end_roi();
    printf("program done\n");

for (int i = 0; i < node_start_id; i++) {
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "predicted_value: " << nodes[i].predicted_value << std::endl;
    std::cout << "num_instances: " << nodes[i].num_instances << std::endl;
    std::cout << "left_child_id: " << nodes[i].left_child_id << std::endl;
    std::cout << "right_child_id: " << nodes[i].right_child_id << std::endl;
    std::cout << "start_index: " << nodes[i].start_index << std::endl;
    std::cout << "feature_id: " << nodes[i].feature_id << std::endl;
    std::cout << "feature_threshold: " << nodes[i].feature_threshold << std::endl;
    std::cout << "split_index: " << nodes[i].split_index << std::endl;
    std::cout << "-------------------------" << std::endl;
}
    cudaFree(d_data);
    cudaFree(d_new_data);
    cudaFree(d_buffer);
    cudaFree(d_split_direction);
    cudaFree(d_label);
    cudaFree(d_key);
    cudaFree(d_nodes);
    cudaFree(d_num_node_next_level);
    cudaFree(d_lock);
}



























