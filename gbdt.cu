#include <iostream>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <sys/time.h>
#include <vector>
#include <set>








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
# define GainLambda 1
// # define numRows 8
// # define numCols 4
# define MinimumSplitNumInstances 1
# define NUM_THREAD 512
# define Gamma 0



class node
{
public:

    VTYPE predicted_value;
    int node_id;
    int num_instances; // number of instances
    int level;
    int left_child_id;
    int right_child_id;
    double training_loss;
    bool is_leaf;
    int start_index; // start index in data
    int feature_id;
    int index_in_segment;  // the feature we use to split the node
    int split_index;  // the feature we use to split the node
    VTYPE feature_threshold; // the threshold of the feature value; if it is larger than threshold, it goes to the right child, otherwise the left child

    __host__ __device__ node()
    {
        predicted_value = -1;
        node_id = -1;
        num_instances = -1; // number of instances
        level = -1;
        split_index = -1;
        left_child_id = -1;
        right_child_id = -1;
        training_loss = 0.0;
        is_leaf = false;
        start_index = 0;
        feature_id = -1;
        feature_threshold = 0.0;
        index_in_segment = -1;
    }
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








// __global__ void tree_GPU(node* d_nodes, attribute_id_pair* d_data, VTYPE* d_label, int* d_num_node_eachlevel,int* d_total_num_nodes )
// {   
//     int num_node_this_level = *d_num_node_eachlevel;
//     int total_num_nodes_this_level = *d_total_num_nodes;
//     int total_num_nodes_next_level = total_num_nodes_this_level;
//     int num_node_next_level = 0;
//     printf("d_nodes[0].num_instances %d\n", d_nodes[0].num_instances);
//     printf("num_node_this_level %d, total_num_nodes_this_level %d\n", num_node_this_level, total_num_nodes_this_level);
//     for (int cur_node_id = total_num_nodes_this_level -  num_node_this_level; cur_node_id < total_num_nodes_this_level; cur_node_id++) {
//             printf("cur_node_id %d, total_num_nodes_this_level %d, (cur_node_id < total_num_nodes_this_level) %d\n", cur_node_id, total_num_nodes_this_level, (cur_node_id < total_num_nodes_this_level));
//             node cur_node = d_nodes[cur_node_id];
//             if (cur_node.num_instances < MinimumSplitNumInstances) {
//                 // printf("cur_node.num_instances %d\n", cur_node.num_instances);
//                 // printf("continue\n");
//                 continue;
//             }
//             int num_instances= cur_node.num_instances;
//             int start_index = cur_node.start_index;
//             int node_size = num_instances* NUM_FEATURE;
//             VTYPE best_gain = 0;
//             VTYPE best_split_point = 0;
//             int best_split_index = -1;
//             int best_split_index_in_segment = 0;
//             VTYPE gamma = 0.0;
//             VTYPE sum_y = 0;
            
//             for (int i = start_index ; i < start_index  + num_instances; i++) {
//                 attribute_id_pair pair = d_data[i];
//                 int instance_id = pair.instance_id;
                
//                 // Access the corresponding label in d_label using the instance_id
//                 VTYPE y = d_label[instance_id];
//                 sum_y += y;
//                 // Perform summation of y or use the value for computation
//                 // ...
//             }
            
//             // Calculate the mean of sum_y as prediction
//             VTYPE prediction = sum_y / num_instances;

//             VTYPE Diff[DataSize];
//             for (int i = start_index ; i < start_index  + node_size; i++) {
//                 attribute_id_pair pair = d_data[i];
//                 int instance_id = pair.instance_id;
                
//                 // Access the corresponding label in d_label using the instance_id
//                 VTYPE y = d_label[instance_id];
//                 Diff[i] = y - prediction;
//                 // Perform summation of y or use the value for computation
//                 // ...
//             }
            
//             // Calculate the prefix sum of Diff
//             VTYPE presum[ DataSize];
//             presum[start_index ] = Diff[start_index ];
//             for (int i = start_index + 1; i < start_index  + node_size; i++) { // ? + node_size?
//                 presum[i] = presum[i-1] + Diff[i];
//             }

//             for (int j = 0; j < node_size / NUM_FEATURE; j++) {
//                 for (int k = 0; k < NUM_FEATURE; k++) {
//                     int instanceId = k * (node_size / NUM_FEATURE) + j;

//                     // Not the last instance in one attribute
//                     if (((instanceId + 1) % NUM_FEATURE) != 0) {
//                         VTYPE split_point = (d_data[instanceId].attribute_value + d_data[instanceId + 1].attribute_value) / 2;
//                         VTYPE G_l = presum[instanceId];
//                         VTYPE G_r = presum[(k + 1) * (node_size / NUM_FEATURE) - 1] - presum[instanceId];
//                         VTYPE H_l = (instanceId + 1 ) * 2;
//                         VTYPE H_r = (node_size + start_index  - instanceId - 1) * 2;
//                         VTYPE gain = 0.5 * (G_l * G_l / (H_l + GainLambda) + G_r * G_r / (H_r + GainLambda) - (G_l + G_r) * (G_l + G_r) / (H_l + H_r - GainLambda));

//                         if ((gain > best_gain) && (gain > gamma)) {

//                             best_gain = gain;
//                             best_split_point = split_point;
//                             best_split_index = j;
//                             best_split_index_in_segment = k;
//                         }
//                     }
//                 }
//             }
//             // printf("1\n");
//             if (best_split_index == -1) {
//                 continue;
//             }

//             VTYPE left_instanceid [InputNum];
//             for (int split_index = 0; split_index < best_split_index + 1; split_index++) {
//                 int left_index = best_split_index_in_segment * (num_instances) + split_index;

//                 attribute_id_pair pair = d_data[left_index];
//                 int original_id = pair.instance_id;
//                 left_instanceid[split_index] = original_id;
//             }

//             // VTYPE counter [DataSize];
//             extern __shared__ int counter [];
//             for (int i = start_index ; i < start_index  + node_size; i++) {
//                 attribute_id_pair pair = d_data[i];
//                 int id = pair.instance_id;
//                 bool found = false;

//                 // Loop through the left indices array
//                 for (int j = 0; j < node_size; j++) {
//                     if (left_instanceid[j] == id) {
//                         found = true;
//                         break;
//                     }
//                 }

//                 // Check if id is found in left_indices
//                 if (found) {
//                     counter[i] = 1;  // Counter for id found in left_indices
//                 } else {
//                     counter[i] = 0;  // Counter for id not found in left_indices
//                 }
//             }
            
//             int getter[2 * NUM_FEATURE] = {0};  // Initialize getter array with 0s
//             int getter_i = 0;

//             for (int i = 0; i < num_instances* NUM_FEATURE; i += num_instances) {
//                 int count_1 = 0;
//                 int count_0 = 0;

//                 for (int j = 0; j < num_instances; j++) {
//                     if (counter[i + j] == 1) {
//                         count_1++;  // Increment occurrence of 1
//                     } else {
//                         count_0++;  // Increment occurrence of 0
//                     }
//                 }

//                 getter[getter_i] = count_1;
//                 getter[getter_i + 1] = count_0;
//                 getter_i += 2;
//             }

//             int getter_group[2 * NUM_FEATURE] = {0};
//             int leftIndex = 0;
//             int rightIndex = NUM_FEATURE;

//             for (int i = 0; i < 2 * NUM_FEATURE; i++) {
//                 if (i % 2 == 0) {
//                     getter_group[leftIndex] = getter[i];
//                     leftIndex++;
//                 } else {
//                     getter_group[rightIndex] = getter[i];
//                     rightIndex++;
//                 }
//             }
            
//             // printf("2\n");
//             int presum_getter[2 * NUM_FEATURE] = {0};  // Initialize presum_getter array with 0s

//             // Calculate prefix sum of getter
//             presum_getter[0] = 0;  // First element is 0
//             for (int i = 1; i < 2 * NUM_FEATURE; i++) {
//                 presum_getter[i] = presum_getter[i - 1] + getter_group[i - 1];
//             }

//             attribute_id_pair sorted_data[DataSize];


//         // Sort data accoring to prefixsum
//             for (int f = 0; f < NUM_FEATURE; f++) {
//                 for (int j = 0; j < num_instances; j++) {
//                     int instanceId = f * num_instances + j;
//                     int offset_left = 0;
//                     int offset_right = 0;
//                     // Check if d_data[instanceId].instance_id is in left_instanceid
//                     bool found = false;
//                     for (int k = 0; k < node_size; k++) {
//                         if (d_data[instanceId].instance_id == left_instanceid[k]) {
//                             found = true;
//                             break;
//                         }
//                     }

//                     // Determine offset and assign values to sorted_data based on found flag
//                     if (found) {
//                         offset_left = presum_getter[f];
//                         sorted_data[offset_left] = d_data[instanceId];
//                         offset_left++;
//                     } else {
//                         offset_right = presum_getter[NUM_FEATURE + f];
//                         sorted_data[offset_right] = d_data[instanceId];
//                         offset_right++;
//                     }
//                 }
//             }
//             memcpy(d_data,sorted_data, sizeof(VTYPE) * DataSize);
//             int new_start_index = -1;

//             // Scan data to find the first element where instance_id is not in left_instanceid
//             for (int i = 0; i < DataSize; i++) {
//                 bool found = false;
//                 for (int j = 0; j < node_size; j++) {
//                     if (d_data[i].instance_id == left_instanceid[j]) {
//                         found = true;
//                         break;
//                     }
//                 }
//                 if (!found) {
//                     new_start_index = i;
//                     break;
//                 }
//             }
//             cur_node.start_index = start_index;
//             cur_node.num_instances = num_instances;
//             // cur_node.level = level ;
//             // cur_node.left_child_id = left_child_id;
//             // cur_node.right_child_id = right_child_id;
//             cur_node.training_loss = best_gain;
//             // cur_node.is_leaf = is_leaf;
//             cur_node.feature_id = best_split_index_in_segment;
//             cur_node.feature_threshold = best_split_point;
//             cur_node.split_index = best_split_index;

//             // Create the left child node
//             node left_child;
           
//             left_child.start_index = start_index;
//             left_child.num_instances = (new_start_index - start_index)/NUM_FEATURE;
//             left_child.node_id = total_num_nodes_next_level;
//             d_nodes[total_num_nodes_next_level] = left_child;

//             cur_node.left_child_id = total_num_nodes_next_level;
            
//             total_num_nodes_next_level++;
        
//             // left_child .level = level + 1;

//             // Create the right child node
//             node right_child;
//             right_child.start_index = new_start_index;
//             right_child.num_instances = (node_size - new_start_index)/ NUM_FEATURE;
//             right_child.node_id = total_num_nodes_next_level;
//             // right_child.level = level + 1;
//             // Assign left and right child nodes to d_nodes array
//             d_nodes[total_num_nodes_next_level] = right_child;
            
//             cur_node.right_child_id = total_num_nodes_next_level;
//             printf("total_num_nodes_next_level %d\n", total_num_nodes_next_level);
//             total_num_nodes_next_level++;

//             d_nodes[cur_node_id] = cur_node;
//         }
//         *d_total_num_nodes = total_num_nodes_next_level; 
//         *d_num_node_eachlevel = (total_num_nodes_next_level - total_num_nodes_this_level);
// }

__global__ void get_gradient(node* d_nodes, attribute_id_pair* d_data, VTYPE* d_label, VTYPE* d_buffer) {
    int node_id = blockDim.x;
    __shared__ node cur_node = d_nodes[node_id];
    int start_index = cur_node.start_index + threadIdx.x;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    int instance_id;
    VTYPE y, y_hat, gradient;

    for (int index = start_index; index < end_index; index += blockDim.x) {
        attribute_id_pair pair = d_data[index];
        instance_id = pair.instance_id;
        y = d_label[instance_id];
        y_hat = cur_node.predicted_value;
        gradient = 2 * (y_hat - y);
        d_buffer[index] = gradient;
    }
}

// a segment is a feature in a node. A feature contains some instances
__global__ void set_key_segment(node* d_nodes, int* d_key) {
    int node_id = blockDim.x;
    __shared__ node cur_node = d_nodes[node_id];
    int start_index = cur_node.start_index + threadIdx.x;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    int key, feature_num;
    

    for (int index = start_index; index < end_index; index += blockDim.x) {
        feature_num = (index / num_instances)
        key = node_id * NUM_FEATURE + feature_num;
        d_key[index] = key;
    }
}


__global__ void get_gain(node* d_nodes, int* d_key, VTYPE* d_buffer) { // d_buffer is prefix sum so far
    int node_id = blockDim.x;
    __shared__ node cur_node = d_nodes[node_id];
    int start_index = cur_node.start_index + threadIdx.x;
    int num_instances = cur_node.num_instances;
    int end_index = start_index + num_instances * NUM_FEATURE;
    VTYPE gain, H_l, H_r, G_l, G_r, G_L_plus_G_r;
    int index_in_segment, last_instance_index;
    int num_left_instance, num_right_instance;

    // skip the last instance
    for (int index = start_index; index < end_index; index += blockDim.x) {
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
        gain = ((G_l * G_l / (H_l + lambda)) + (G_r * G_r / (H_r + lambda)) - (G_L_plus_G_r * G_L_plus_G_r / (H_l + H_r - lambda))) * 0.5;
        d_buffer[index] = gain;
    }

    __syncthreads();

    // write 0 into last instance
    start_index = cur_node.start_index + (num_instances * threadIdx.x);
    int increment = num_instances * blockDim.x;
    for (int index = start_index; index < end_index; index += increment) {
        d_buffer[index] = 0;
    }
}

__global__ void get_best_split_point(node* d_nodes, VTYPE* d_buffer, int* d_best_split_index) {
    int node_id = blockDim.x;
    __shared__ node cur_node = d_nodes[node_id];
    int thread_idx = threadIdx.x
    int index_of_this_thread = cur_node.start_index + thread_idx;
    int num_instances = cur_node.num_instances;
    int end_index = index_of_this_thread + num_instances * NUM_FEATURE;
    VTYPE max_value = 0;
    VTYPE compared_value;
    int max_index = -1;
    for (int index = index_of_this_thread; index < end_index; index += blockDim.x) {
        compared_value = d_buffer[index];
        if (compared_value > max_value) {
            max_value = compared_value;
            max_index = index;
        }
    }
    int node_size = cur_node.node_size;
    int last_num_active_thread = node_size;
    int num_active_thread = (node_size + 1) / 2;
    int compared_thread_idx, compared_index;
    while (last_num_active_thread > 1) {
        if (thread_idx >= num_active_thread && thread_idx < last_num_active_thread) {
            d_buffer[index_of_this_thread] = max_value;
            d_best_split_index[index_of_this_thread] = max_index;
        }
        __syncthreads();
        compared_thread_idx = thread_idx + num_active_thread;
        compared_index = cur_node.start_index + compared_thread_idx;
        if (thread_idx < num_active_thread && compared_thread_idx < last_num_active_thread) {
            compared_value = d_buffer[compared_index];
            if (compared_value > max_value) {
                max_value = compared_value;
                max_index = compared_index;
            }
        }
        last_num_active_thread = num_active_thread;
        num_active_thread = (num_active_thread + 1) / 2;
    }

    if (thread_idx == 0 && max_value > Gamma) {
        d_nodes[node_id].split_index = max_index;
    }
}
int main(void) {
    node nodes[MaxNodeNum];
    node root;
    root.node_id = 0;
    root.num_instances = InputNum;
    root.level = 0;
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
    cudaMalloc((void **)(&d_data), sizeof(VTYPE) * DataSize);
    cudaMemcpy(d_data, data, sizeof(VTYPE) * DataSize, cudaMemcpyHostToDevice);

    int *d_key;
    cudaMalloc((void **)(&d_key), sizeof(int) * DataSize);

    VTYPE *d_buffer;
    cudaMalloc((void **)(&d_buffer), sizeof(VTYPE) * DataSize);

    
    int *d_best_split_index;
    cudaMalloc((void **)(&d_best_split_index), sizeof(int) * DataSize);

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
    int *d_num_node_eachlevel;
    int value = 1;
    int *num_node_this_level = &value;
    cudaMalloc((int **)(&d_num_node_eachlevel), sizeof(int)); // Use d_label instead of label
    cudaMemcpy(d_num_node_eachlevel, num_node_this_level, sizeof(int), cudaMemcpyHostToDevice);

    int *d_total_num_nodes;
    int *total_num_nodes = &value;
    cudaMalloc((int **)(&d_total_num_nodes), sizeof(int)); // Use d_label instead of label
    cudaMemcpy(d_total_num_nodes, total_num_nodes, sizeof(int), cudaMemcpyHostToDevice);

    

    dim3 block_size, thread_size;
    while (level < MaxDepth - 1) {
        printf("level %d\n", level);
        block_size = (*num_node_this_level);
`       thread_size = (NUM_THREAD);
        get_gradient<<<block_size, thread_size>>>(d_nodes, d_data, d_label, d_buffer);
        set_key_segment<<<block_size, thread_size>>>(d_node, d_key);
        thrust::inclusive_scan_by_key(thrust::system::cuda::par, d_key, d_key + DataSize, d_buffer, d_buffer); // find G
        get_gain<<<block_size, thread_size>>>(d_nodes, d_key, d_buffer);
        // it is inclusive so for split point at index i, data i belong to the left child
        get_best_split_point<<<block_size, thread_size>>>(d_nodes, d_buffer, d_best_split_index);
        cudaDeviceSynchronize();
        cudaMemcpy(data, d_data, sizeof(attribute_id_pair)* DataSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(nodes, d_nodes, sizeof(node)* MaxNodeNum, cudaMemcpyDeviceToHost);
        cudaMemcpy(num_node_this_level, d_num_node_eachlevel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(total_num_nodes, d_total_num_nodes, sizeof(int), cudaMemcpyDeviceToHost);
        level++;
    }

    // begin_roi();
    // classifier_GPU<<<grid_size, block_size>>>(d_data);
    // cudaDeviceSynchronize();
    // end_roi();

for (int i = 0; i < MaxNodeNum; i++) {
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "predicted_value: " << nodes[i].predicted_value << std::endl;
    std::cout << "node_id: " << nodes[i].node_id << std::endl;
    std::cout << "num_instances: " << nodes[i].num_instances << std::endl;
    std::cout << "level: " << nodes[i].level << std::endl;
    std::cout << "left_child_id: " << nodes[i].left_child_id << std::endl;
    std::cout << "right_child_id: " << nodes[i].right_child_id << std::endl;
    std::cout << "training_loss: " << nodes[i].training_loss << std::endl;
    std::cout << "is_leaf: " << nodes[i].is_leaf << std::endl;
    std::cout << "start_index: " << nodes[i].start_index << std::endl;
    std::cout << "feature_id: " << nodes[i].feature_id << std::endl;
    std::cout << "feature_threshold: " << nodes[i].feature_threshold << std::endl;
    std::cout << "split_index: " << nodes[i].split_index << std::endl;
    std::cout << "-------------------------" << std::endl;
}
    cudaFree(d_nodes);
    cudaFree(d_data);
    cudaFree(d_label);
}



























