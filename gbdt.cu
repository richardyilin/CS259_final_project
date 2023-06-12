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








using namespace std;








//Define the parameters if not defined externally
#ifndef cmd_def
#define InputNum 2  // Number of input data points (instances)
#define NUM_FEATURE 2  // Number of features in an instance
#define MaxDepth 2  // Number of features in an instance
# define MaxNodeNum (static_cast<int>(pow(2, MaxDepth)) - 1)
#endif
#define VTYPE float
# define DataSize (NUM_FEATURE * InputNum)
# define Lambda 1
// # define numRows 8
// # define numCols 4
# define MinimumSplitNumInstances 1
# define NUM_THREAD 512
# define Gamma 0
# define Left true
# define Right false
# define KB (1 << 3)
# define MB (1 << 6)
# define GB (1 << 9)
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
    for (int i = 0; i < NUM_FEATURE; i++) {
        for (int j = 0; j < InputNum; j++) {
            attribute_id_pair pair;
            int id = i * InputNum + j;
            pair.attribute_value = VTYPE(id);
            pair.instance_id = id;
            data[id] = pair;
        }
    }

    for (int i = 0; i < DataSize; i++){
        label[i] = i;
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


__global__ void get_gain(node* d_nodes, int* d_key, VTYPE* d_buffer, int node_start_id) { // d_buffer is prefix sum so far
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    int start_index_of_this_thread = cur_node.start_index + thread_idx;
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    VTYPE gain, H_l, H_r, G_l, G_r, G_L_plus_G_r;
    int index_in_segment, last_instance_index;
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
        last_instance_index = (((index / num_instances) + 1) * num_instances) - 1;
        G_l = d_buffer[index];
        G_r = d_buffer[last_instance_index] - G_l;
        H_l = (index_in_segment + 1) * 2;
        H_r = (num_instances - index_in_segment) * 2;
        G_L_plus_G_r = G_l + G_r;
        gain = ((G_l * G_l / (H_l + Lambda)) + (G_r * G_r / (H_r + Lambda)) - (G_L_plus_G_r * G_L_plus_G_r / (H_l + H_r - Lambda))) * 0.5;
        d_buffer[index] = gain;
    }

    __syncthreads();

    // write 0 into last instance
    int loop_start_index = (cur_node.start_index + (num_instances * thread_idx) - 1);
    int increment = num_instances * blockDim.x;
    for (int index = loop_start_index; index < end_index; index += increment) {
        d_buffer[index] = 0;
    }
}

__global__ void get_best_split_point(node* d_nodes, VTYPE* d_buffer, int node_start_id, attribute_id_pair* d_data) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
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
    }
    extern __shared__ VTYPE max_values[];
    extern __shared__ int max_indices[];

    int last_num_active_thread = num_thread_per_block;
    int num_active_thread = (last_num_active_thread + 1) / 2;
    int compared_thread_idx, compared_index;
    while (last_num_active_thread > 1) {
        if (thread_idx >= num_active_thread && thread_idx < last_num_active_thread) {
            max_values[start_index_of_this_thread] = max_value;
            max_indices[start_index_of_this_thread] = max_index;
        }
        __syncthreads();
        compared_thread_idx = thread_idx + num_active_thread;
        compared_index = cur_node.start_index + compared_thread_idx;
        if (thread_idx < num_active_thread && compared_thread_idx < last_num_active_thread) {
            compared_value = max_values[compared_index];
            if (compared_value > max_value) {
                max_value = compared_value;
                max_index = compared_index;
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
        } else {
            d_nodes[node_id].split_index = -1;
            d_nodes[node_id].feature_id = -1;
            d_nodes[node_id].feature_threshold = 0.0;
        }
    }
}
__global__ void set_counter(node* d_nodes, VTYPE* d_buffer, int* d_counter, int * d_key, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data) {
    // the range of counter is [2*blockDim.x:4*BlocDim.x]
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
    int split_index = cur_node.split_index;
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    int node_size = num_instances * NUM_FEATURE;
    // int num_thread_per_block = blockDim.x;
    int num_thread_per_block = min(blockDim.x, node_size);
    int left_counter_index = 2 * blockDim.x + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    // reset key and counter
    for (int index = left_counter_index; index <= right_counter_index; index += num_thread_per_block) { // can move into if
        d_key[index] = -1;
        d_counter[index] = 0; // may not need this
    }
    if (split_index == -1) { // the entire node does not need to split
        return;
    }
    if (thread_idx >= node_size) {
        return;
    }
    int local_counter[2] = {0, 0};

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    int feature_id = d_nodes[node_id].feature_id;
    int split_start_index = start_index + (feature_id * num_instances);
    int end_index = start_index_of_this_thread + num_instances_per_thread;
    int instance_id;
    int left_instance_id;
    bool go_left = false;
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
}

__global__ void split_node(node* d_nodes, int* d_counter, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data, int* d_lock, int* d_num_node_next_level, attribute_id_pair* d_new_data, int num_node_cur_level) {
    int node_id = blockIdx.x + (node_start_id);
    int thread_idx = threadIdx.x;
    __shared__ node cur_node;
    if (thread_idx == 0) {
        cur_node = d_nodes[node_id];
    }
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
    // int num_thread_per_block = blockDim.x;
    int num_thread_per_block = min(blockDim.x, node_size);
    int left_counter_index = 2 * blockDim.x + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    int local_counter[2];
    local_counter[0] = d_counter[left_counter_index];
    local_counter[1] = d_counter[right_counter_index];

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    int feature_id = d_nodes[node_id].feature_id;
    int split_start_index = start_index + (feature_id * num_instances);
    int end_index = start_index_of_this_thread + num_instances_per_thread;
    int instance_id;
    int left_instance_id;
    bool direction;
    int new_index;
    attribute_id_pair cur_data;
    // move data
    for (int index = start_index_of_this_thread; index < end_index; index++) {
        direction = d_split_direction[index];
        if (direction == Left) {
            new_index = local_counter[0];
            local_counter[0] ++;
        } else {
            new_index = local_counter[1];
            local_counter[1] ++;
        }
        cur_data = d_data[index];
        d_new_data[new_index] = cur_data;
    }

    // create two new nodes

    if (thread_idx == 0) {
        int next_node_start_id = node_start_id + num_node_cur_level;
        int left_child_id = next_node_start_id + num_node_next_level + 1;
        int right_child_id = left_child_id + 1;
        cur_node.left_child_id = left_child_id;
        cur_node.right_child_id = right_child_id;
        d_nodes[node_id] = cur_node;

        node left_child;
        int left_child_num_instance = (split_index - start_index) % num_instances;
        left_child.num_instances = left_child_num_instance;
        left_child.start_index = start_index;
        d_nodes[left_child_id] = left_child;

        node right_child;
        right_child.num_instances = num_instances - left_child_num_instance;
        right_child.start_index = start_index + (left_child_num_instance * NUM_FEATURE);
        d_nodes[right_child_id] = right_child;
    }
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
    VTYPE label [DataSize];
    read_input(data, label);
    cout << "starting program\n";
    // fill_data(data);
    // fill_label(label);


    attribute_id_pair *d_data;
    cudaMalloc((void **)(&d_data), sizeof(attribute_id_pair) * DataSize);
    cudaMemcpy(d_data, data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyHostToDevice);
    attribute_id_pair *d_new_data;
    cudaMalloc((void **)(&d_new_data), sizeof(attribute_id_pair) * DataSize);

    int *d_key;
    cudaMalloc((void **)(&d_key), sizeof(int) * DataSize);

    VTYPE *d_buffer;
    cudaMalloc((void **)(&d_buffer), sizeof(VTYPE) * DataSize);

    
    // int *d_best_split_index;
    // cudaMalloc((void **)(&d_best_split_index), sizeof(int) * DataSize);

    
    bool *d_split_direction;
    cudaMalloc((void **)(&d_split_direction), sizeof(bool) * DataSize);

    VTYPE *d_label;
    cudaMalloc((void **)(&d_label), sizeof(VTYPE) * DataSize); // Use d_label instead of label
    cudaMemcpy(d_label, label, sizeof(VTYPE) * DataSize, cudaMemcpyHostToDevice);

    // dim3 block_size(512); //test, may change in different levels in a tree, calculated by a formula
    // dim3 grid_size(16, 16); //test, may change in different levels in a tree, calculated by a formula
    // Allocate memory on the GPU
    node *d_nodes;
    cudaMalloc((void**)&d_nodes, sizeof(node) * MaxNodeNum);
    // Copy the nodes array to the GPU
    cudaMemcpy(d_nodes, nodes, sizeof(node) * MaxNodeNum, cudaMemcpyHostToDevice);
    


 

    int level = 0;

    int *d_num_node_next_level;
    cudaMalloc((int **)(&d_num_node_next_level), sizeof(int)); // Use d_label instead of label

    // int *d_total_num_nodes;
    // cudaMalloc((int **)(&d_total_num_nodes), sizeof(int)); // Use d_label instead of label

    const int zero = 0;
    int* d_lock;
    cudaMalloc((int **)(&d_lock), sizeof(int)); // Use d_label instead of label
    cudaMemcpy(d_lock, &zero, sizeof(int), cudaMemcpyHostToDevice);

    

    int num_node_cur_level = 1;
    int node_start_id = 0;
    dim3 block_size(num_node_cur_level, 0, 0);
    dim3 thread_size(NUM_THREAD, 0, 0);
    int num_cache_entry_per_block;
    int dynamic_memory_size;
    while (level < MaxDepth - 1) {
        // printf("level %d\n", level);
        cudaMemcpy(d_num_node_next_level, &zero, sizeof(int), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_total_num_nodes, &total_num_nodes, sizeof(int), cudaMemcpyHostToDevice);
        block_size.x = num_node_cur_level;
        get_gradient<<<block_size, thread_size>>>(d_nodes, d_data, d_label, d_buffer, node_start_id);
        set_key_segment<<<block_size, thread_size>>>(d_nodes, d_key, node_start_id);
        thrust::inclusive_scan_by_key(thrust::system::cuda::par, d_key, d_key + DataSize, d_buffer, d_buffer); // find G
        get_gain<<<block_size, thread_size>>>(d_nodes, d_key, d_buffer, node_start_id);
        // it is inclusive so for split point at index i, data i belong to the left child
        num_cache_entry_per_block = DynamicMemorySize / (sizeof(VTYPE) + sizeof(int)) / (num_node_cur_level);
        thread_size.x = min(num_cache_entry_per_block, NUM_THREAD);
        dynamic_memory_size = thread_size.x * (sizeof(VTYPE) + sizeof(int));
        get_best_split_point<<<block_size, thread_size, dynamic_memory_size>>>(d_nodes, d_buffer, node_start_id, d_data);
        num_cache_entry_per_block = DynamicMemorySize / sizeof(int) / (num_node_cur_level) / 2;
        thread_size.x = min(num_cache_entry_per_block, NUM_THREAD);        
        int *d_counter;
        cudaMalloc((void **)(&d_counter), sizeof(int) * thread_size.x * (num_node_cur_level) * 2);
        set_counter<<<block_size, thread_size>>>(d_nodes, d_buffer, d_counter, d_key, node_start_id, d_split_direction, d_data);
        thrust::exclusive_scan_by_key(thrust::system::cuda::par, d_key, d_key + (thread_size.x * 2), d_counter, d_counter); // find G
        split_node<<<block_size, thread_size>>>(d_nodes, d_counter, node_start_id, d_split_direction, d_data, d_lock, d_num_node_next_level, d_new_data, num_node_cur_level);
        cudaFree(d_counter);
        cudaDeviceSynchronize();
        // cudaMemcpy(data, d_data, sizeof(attribute_id_pair)* DataSize, cudaMemcpyDeviceToHost);
        // cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
        cudaMemcpy(&num_node_cur_level, d_num_node_next_level, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&total_num_nodes, node_start_id, sizeof(int), cudaMemcpyDeviceToHost);
        node_start_id += num_node_cur_level;
        level++;
    }

    cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
    // begin_roi();
    // classifier_GPU<<<grid_size, block_size>>>(d_data);
    // cudaDeviceSynchronize();
    // end_roi();

for (int i = 0; i < MaxNodeNum; i++) {
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "predicted_value: " << nodes[i].predicted_value << std::endl;
    std::cout << "num_instances: " << nodes[i].num_instances << std::endl;
    std::cout << "right_child_id: " << nodes[i].right_child_id << std::endl;
    std::cout << "start_index: " << nodes[i].start_index << std::endl;
    std::cout << "feature_id: " << nodes[i].feature_id << std::endl;
    std::cout << "feature_threshold: " << nodes[i].feature_threshold << std::endl;
    std::cout << "split_index: " << nodes[i].split_index << std::endl;
    std::cout << "-------------------------" << std::endl;
}
    cudaFree(d_nodes);
    cudaFree(d_data);
    cudaFree(d_label);
    cudaFree(d_buffer);
    cudaFree(d_key);
    cudaFree(d_num_node_next_level);
    // cudaFree(d_total_num_nodes);
}



























