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
#include <thrust/execution_policy.h>
#include <functional>
#include <algorithm>
#include <random>
#include <array>

using namespace std;

//Define the parameters if not defined externally
#ifndef cmd_def
#define InputNum 45730  // Number of input data points (instances)
#define NUM_FEATURE 9  // Number of features in an instance
# define MinimumSplitNumInstances 3
#define MaxDepth 10  // Number of features in an instance
# define MaxNodeNum (static_cast<int>(pow(2, MaxDepth)) - 1) // maximal number of nodes, can change to any positive integer
#endif
#define VTYPE float
# define DataSize (NUM_FEATURE * InputNum)
# define Lambda 1
# define NUM_THREAD 512
# define Gamma 0
# define Left true
# define Right false
# define KB (1 << 10)
# define MB (1 << 20)
# define GB (1 << 30)
# define DynamicMemorySize (1 * GB)
# define ATTRIBUTE_VALUE_LOWER_BOUND -1000000
# define ATTRIBUTE_VALUE_UPPER_BOUND 1000000


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
};



#include <utility>


class attribute_id_pair {
public:
   VTYPE attribute_value;
   int instance_id;
};

inline void cuda_check_error() {
    auto err = cudaGetLastError();
    if(err) {
      cerr << "Error: " << cudaGetErrorString(err) << endl;
      exit(0);
    }
  }

VTYPE generateRandomNumber(VTYPE lower_bound, VTYPE upper_bound) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<VTYPE> distribution(lower_bound, upper_bound);
    VTYPE random_number = distribution(gen);
    return random_number;
}

void fill_data(attribute_id_pair* data) {
    for (int index = 0; index < DataSize; index++) {
        attribute_id_pair pair;
        pair.attribute_value = generateRandomNumber(ATTRIBUTE_VALUE_LOWER_BOUND, ATTRIBUTE_VALUE_UPPER_BOUND);
        pair.instance_id = index % InputNum;
        data[index] = pair;
    }
}
bool compareByAttributeValue(const attribute_id_pair& pair1, const attribute_id_pair& pair2) {
    return pair1.attribute_value < pair2.attribute_value;
}

// Function to sort the array
void sortArrayByAttributeValue(attribute_id_pair* data) {
    for (int i = 0; i < DataSize; i+=InputNum) {
        std::sort(data + i, data + (i + InputNum), compareByAttributeValue);
    }
}

void fill_label(VTYPE* label){
    for (int i = 0; i < InputNum; i++){
        label[i] = generateRandomNumber(ATTRIBUTE_VALUE_LOWER_BOUND, ATTRIBUTE_VALUE_UPPER_BOUND);
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
 cout << "elapsed (sec): " << usec/1000000.0 << "\n";
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

    VTYPE gain, H_l, H_r, G_l, G_r, G_L_plus_G_r;
    int index_in_segment, last_local_data_index;
    int num_left_instance, num_right_instance;

    // skip the last instance
    for (int index = start_index_of_this_thread; index < end_index; index += blockDim.x) {
        index_in_segment = (index - start_index) % num_instances;
        if (index_in_segment == NUM_FEATURE - 1) { // the last instance in the feature, cannot split because the right child has no instances
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
    }
    extern __shared__ VTYPE shared_mem[];
    VTYPE* max_values = reinterpret_cast<VTYPE*>(shared_mem);
    int* max_indices = reinterpret_cast<int*>(shared_mem + num_thread_per_block);

    int last_num_active_thread = num_thread_per_block;
    int num_active_thread = (last_num_active_thread + 1) / 2;
    int compared_thread_idx;
    while (last_num_active_thread > 1) {
        if (thread_idx >= num_active_thread && thread_idx < last_num_active_thread) {
            max_values[thread_idx] = max_value;
            max_indices[thread_idx] = max_index;
        }
        __syncthreads();

        compared_thread_idx = thread_idx + num_active_thread;
        if (thread_idx < num_active_thread && compared_thread_idx < last_num_active_thread) {
            compared_value = max_values[compared_thread_idx];
            if (compared_value > max_value) {
                max_value = compared_value;
                max_index = max_indices[compared_thread_idx];
            }
        }
        last_num_active_thread = num_active_thread;
        num_active_thread = (num_active_thread + 1) / 2;
    }

    if (thread_idx == 0) {
        if (max_value > Gamma && max_index != -1) {
            d_nodes[node_id].split_index = max_index;
            int feature_id = (max_index - start_index) / num_instances;
            d_nodes[node_id].feature_id = feature_id;
            VTYPE feature_threshold = (d_data[max_index].attribute_value + d_data[max_index+1].attribute_value) / 2;
            d_nodes[node_id].feature_threshold = feature_threshold;
        }
    }
}
__global__ void set_counter(node* d_nodes, VTYPE* d_buffer, int* d_counter, int * d_key, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data) {
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
    int num_thread_per_block = min(blockDim.x, node_size);
    int global_counter_start_index = 2 * (min(start_index, blockIdx.x * blockDim.x)); // * 2 because each thread has two counters
    int left_counter_index = global_counter_start_index + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    // set key
    for (int index = left_counter_index; index <= right_counter_index; index += num_thread_per_block) { // can move into if
        d_key[index] = node_id;
    }
    int local_counter[2] = {0, 0};

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    int feature_id = cur_node.feature_id;
    int split_start_index = start_index + (feature_id * num_instances);
    int end_index = min(start_index_of_this_thread + num_instances_per_thread, start_index + node_size);
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
__global__ void creat_child_nodes(node* d_nodes, int node_start_id, int* d_lock, int* d_num_node_next_level, int num_node_cur_level) {
    int node_id = blockIdx.x + (node_start_id);
    node cur_node = d_nodes[node_id];
    int split_index = cur_node.split_index;
    int start_index = cur_node.start_index;
    int num_instances = cur_node.num_instances;
    if (split_index == -1) {
        return;
    }
    // get child node id
    int num_node_next_level;
    int next_node_start_id = node_start_id + num_node_cur_level;
    int right_child_id;
    bool wait_lock = true;
    while (wait_lock) {
        if(atomicCAS(d_lock, 0, 1) == 0){
            num_node_next_level = *d_num_node_next_level;
            right_child_id = next_node_start_id + num_node_next_level + 1;
            if (right_child_id < MaxNodeNum) {
                atomicAdd(d_num_node_next_level, 2);
            }
            atomicExch(d_lock, 0);
            wait_lock = false;
        }
    }
    if (right_child_id < MaxNodeNum) {
        int left_child_id = right_child_id - 1;
        d_nodes[node_id].left_child_id = left_child_id;
        d_nodes[node_id].right_child_id = right_child_id;

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
    } else { // exceed max node number so do not split
        d_nodes[node_id].split_index = -1;
    }
}
__global__ void split_node(node* d_nodes, int* d_counter, int node_start_id, bool *d_split_direction, attribute_id_pair* d_data, attribute_id_pair* d_new_data) {
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
    int global_counter_start_index = 2 * (min(start_index, blockIdx.x * blockDim.x)); // * 2 because each thread has two counters
    int left_counter_index = global_counter_start_index + thread_idx;
    int right_counter_index = left_counter_index + num_thread_per_block;
    int local_counter[2];
    local_counter[0] = d_counter[left_counter_index];
    local_counter[1] = d_counter[right_counter_index];

    // find split direction
    int num_instances_per_thread = (node_size + num_thread_per_block - 1) / num_thread_per_block;
    int start_index_of_this_thread = start_index + (thread_idx * num_instances_per_thread);
    int end_index = min(start_index_of_this_thread + num_instances_per_thread, start_index + node_size);
    bool direction;
    int new_index;
    attribute_id_pair cur_data;
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
    }
}

__global__ void get_prediciton_value(node* d_nodes, int node_start_id, attribute_id_pair* d_data, VTYPE* d_label) {
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
    if (thread_idx >= num_instances) {
        return;
    }
    int num_thread_per_block = min(blockDim.x, num_instances);
    int end_index = start_index + num_instances;
    VTYPE sum = 0;
    VTYPE addend;
    int instance_id;
    for (int index = start_index_of_this_thread; index < end_index; index += num_thread_per_block) {
        instance_id = d_data[index].instance_id;
        addend = d_label[instance_id];
        sum += addend;
    }
    extern __shared__ VTYPE shared_mem[];

    int last_num_active_thread = num_thread_per_block;
    int num_active_thread = (last_num_active_thread + 1) / 2;
    int addend_thread_idx;
    while (last_num_active_thread > 1) {
        if (thread_idx >= num_active_thread && thread_idx < last_num_active_thread) {
            shared_mem[thread_idx] = sum;
        }
        __syncthreads();
        
        addend_thread_idx = thread_idx + num_active_thread;
        if (thread_idx < num_active_thread && addend_thread_idx < last_num_active_thread) {
            addend = shared_mem[addend_thread_idx];
            sum += addend;
        }
        last_num_active_thread = num_active_thread;
        num_active_thread = (num_active_thread + 1) / 2;
    }

    if (thread_idx == 0) {
        VTYPE predicted_value = sum / num_instances;
        d_nodes[node_id].predicted_value = predicted_value;
    }

}
int main(void) {
    node nodes[MaxNodeNum];
    node root;
    root.num_instances = InputNum;
    root.start_index = 0;
    nodes[0] = root;
    attribute_id_pair data[DataSize];
    VTYPE label [InputNum];
    fill_data(data);
    fill_label(label);
    sortArrayByAttributeValue(data);

    cout << "starting program\n";
    
    begin_roi(); // we assume the input data is sorted


    attribute_id_pair *d_data;
    cudaMalloc((void **)(&d_data), sizeof(attribute_id_pair) * DataSize);
    cudaMemcpy(d_data, data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyHostToDevice);
    attribute_id_pair *d_new_data;
    cudaMalloc((void **)(&d_new_data), sizeof(attribute_id_pair) * DataSize);

    int *d_key;
    cudaMalloc((void **)(&d_key), sizeof(int) * DataSize);

    VTYPE *d_buffer;
    cudaMalloc((void **)(&d_buffer), sizeof(VTYPE) * DataSize);

    bool *d_split_direction;
    cudaMalloc((void **)(&d_split_direction), sizeof(bool) * DataSize);

    VTYPE *d_label;
    cudaMalloc((void **)(&d_label), sizeof(VTYPE) * InputNum);
    cudaMemcpy(d_label, label, sizeof(VTYPE) * InputNum, cudaMemcpyHostToDevice);

    node *d_nodes;
    cudaMalloc((void**)&d_nodes, sizeof(node) * MaxNodeNum);
    cudaMemcpy(d_nodes, nodes, sizeof(node) * MaxNodeNum, cudaMemcpyHostToDevice);

    int level = 0;
    int *d_num_node_next_level;
    cudaMalloc((int **)(&d_num_node_next_level), sizeof(int));
    int* d_lock;
    cudaMalloc((int **)(&d_lock), sizeof(int));
    cudaMemset(d_lock, 0, sizeof(int));
    cuda_check_error();

    int num_node_cur_level = 1;
    int node_start_id = 0;
    dim3 grid_size(num_node_cur_level, 0, 0);
    dim3 block_size(NUM_THREAD, 0, 0);
    int num_cache_entry_per_block;
    int dynamic_memory_size;
    while (level < MaxDepth - 1 && (num_node_cur_level > 0) && node_start_id < MaxNodeNum - 1) {
        cudaMemset(d_num_node_next_level, 0, sizeof(int));
        grid_size = dim3(num_node_cur_level);
        num_cache_entry_per_block = DynamicMemorySize / sizeof(VTYPE) / (num_node_cur_level);
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        dynamic_memory_size = block_size.x * sizeof(VTYPE);
        get_prediciton_value<<<grid_size, block_size, dynamic_memory_size>>>(d_nodes, node_start_id, d_data, d_label);
        cuda_check_error();
        grid_size = dim3(num_node_cur_level);
        block_size = dim3(NUM_THREAD);
        get_gradient<<<grid_size, block_size>>>(d_nodes, d_data, d_label, d_buffer, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        set_key_segment<<<grid_size, block_size>>>(d_nodes, d_key, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        thrust::inclusive_scan_by_key(thrust::device, d_key, d_key + DataSize, d_buffer, d_buffer); // find G
        cudaDeviceSynchronize();
        cuda_check_error();
        get_gain<<<grid_size, block_size>>>(d_nodes, d_buffer, node_start_id);
        cudaDeviceSynchronize();
        cuda_check_error();
        // it is inclusive so for split point at index i, data i belong to the left child
        num_cache_entry_per_block = DynamicMemorySize / (sizeof(VTYPE) + sizeof(int)) / (num_node_cur_level);
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        dynamic_memory_size = block_size.x * (sizeof(VTYPE) + sizeof(int));
        get_best_split_point<<<grid_size, block_size, dynamic_memory_size>>>(d_nodes, d_buffer, node_start_id, d_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        num_cache_entry_per_block = DynamicMemorySize / sizeof(int) / (num_node_cur_level) / 2;
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        int *d_counter;
        cudaMalloc((void **)(&d_counter), sizeof(int) * block_size.x * (num_node_cur_level) * 2);
        cudaDeviceSynchronize();
        cuda_check_error();
        cudaMemset(d_key, -1, sizeof(int) * DataSize);
        cudaDeviceSynchronize();
        cuda_check_error();
        set_counter<<<grid_size, block_size>>>(d_nodes, d_buffer, d_counter, d_key, node_start_id, d_split_direction, d_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        thrust::exclusive_scan_by_key(thrust::device, d_key, d_key + block_size.x * (num_node_cur_level) * 2, d_counter, d_counter);
        cudaDeviceSynchronize();
        cuda_check_error();
        creat_child_nodes<<<grid_size, 1>>>(d_nodes, node_start_id, d_lock, d_num_node_next_level, num_node_cur_level);
        cudaDeviceSynchronize();
        cuda_check_error();
        split_node<<<grid_size, block_size>>>(d_nodes, d_counter, node_start_id, d_split_direction, d_data, d_new_data);
        cudaDeviceSynchronize();
        cuda_check_error();
        cudaMemcpy(d_data, d_new_data, sizeof(attribute_id_pair) * DataSize, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        cuda_check_error();
        cudaFree(d_counter);
        cudaDeviceSynchronize();
        cuda_check_error();
        node_start_id += num_node_cur_level;
        level++;
        cudaMemcpy(&num_node_cur_level, d_num_node_next_level, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cuda_check_error();
    }
    if (num_node_cur_level > 0) {
        grid_size = dim3(num_node_cur_level);
        num_cache_entry_per_block = DynamicMemorySize / sizeof(VTYPE) / (num_node_cur_level);
        block_size = dim3(min(num_cache_entry_per_block, NUM_THREAD));
        dynamic_memory_size = block_size.x * sizeof(VTYPE);
        get_prediciton_value<<<grid_size, block_size, dynamic_memory_size>>>(d_nodes, node_start_id, d_data, d_label);
        cudaDeviceSynchronize();
        cuda_check_error();
        node_start_id += num_node_cur_level;
    }
    cudaMemcpy(nodes, d_nodes, sizeof(node)* node_start_id, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cuda_check_error();
    end_roi();
    printf("program done\n");

for (int i = 0; i < node_start_id; i++) {
    cout << "Node " << i << ":" << endl;
    cout << "predicted_value: " << nodes[i].predicted_value << endl;
    cout << "num_instances: " << nodes[i].num_instances << endl;
    cout << "left_child_id: " << nodes[i].left_child_id << endl;
    cout << "right_child_id: " << nodes[i].right_child_id << endl;
    cout << "start_index: " << nodes[i].start_index << endl;
    cout << "feature_id: " << nodes[i].feature_id << endl;
    cout << "feature_threshold: " << nodes[i].feature_threshold << endl;
    cout << "split_index: " << nodes[i].split_index << endl;
    cout << "-------------------------" << endl;
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



























