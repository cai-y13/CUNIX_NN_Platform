#include <iostream>
#include <cmath>
#include <fstream>

float* LoadConv(const int kernel_size, const int in_channel, const int out_channel, char* path) {
    ifstream weights_file;
    weights_file.open(path);
    int weights_size = kernel_size * kernel_size * in_channel * out_channel;
    float* weights = new float[weights_size];
    for( int i = 0; i < weights_size; i++ ) {
        weights_file >> weights;
        weights++;
    }
    weights_file.close();
    return weights;
}

void Convolution(const int kernel_size, const int pad, const int stride, 
         const float* weights, const float* inputs, float* outputs ) {
    
}

float* LoadFC(const int in_neuron, const int out_neuron, char* path) {
    //Uncompleted
    ifstream weights_file;
    weights_file.open(path);
    int weights_size = in_neuron * out_neuron;
    float* weights = new float[weights_size];
    for( int i = 0; i < weights_size; i++ ) {
        weights_file >> weights;
        weights++;
    }
    weights_file.close()
    return weights;
}

void FullyConnected(const float* weights, const float* inputs, float* outputs) {
    //Uncompleted
    
}

void ReLU(const float* inputs, const int dim, float* outputs) {
    for( int i = 0; i < dim; i++ ) {
        outputs[i] = std::max(0, inputs[i]);
    }
}

void MaxPooling(const float* inputs, const int kernel_size, const int stride, float* outputs, vector<int> shape) {
    int channel = shape[0];
    int height = shape[1];
    int width = shape[2];
    int out_height = static_cast<int>(ceil(static_cast<float>(height - kernel_size) / stride)); 
    int out_width = static_cast<int>(ceil(static_cast<float>(width - kernel_size) / stride)); 
    for( int c = 0; c < channel; c++ ) {
        int offset = c * height * width;
        for( int h = 0; h < out_height; h++ ) {
            for( int w = 0; w < out_width; w++ ) {
                int h_in = h * stride;
                int w_in = w * stride;
                float max = inputs[offset + h_in * width + w_in];
                for( int kh = 0; kh < kernel_size; kh++ ) {
                    for( int kw = 0; kw < kernel_size; kw++ ) {
                        if( inputs[offset + (h_in + kh) * width + w_in + kw] > max) {
                            max = inputs[(h_in + kh) * width + w_in +kw];
                        }
                    }
                }
                outputs[c * out_height * out_width + h * out_width + w] = max;
            }
        } 
    }
}

void Softmax(const float* inputs, const int dim, float* outputs) {
    //Uncompleted
    float scale = 0;
    for( int i = 0; i < dim; i++ ) {
        scale +=  exp(inputs[i]);
    } 
    for( int i = 0; i < dim; i++ ) {
        outputs[i] = exp(inputs[i]) / scale;
    }
}

void im2col(const float* inputs, float* outputs) {
    //Uncompleted
}

int main() {
    float* Conv1 = LoadConv(3, 1, 1, path);
    float* Conv2 = LoadConv(3, 1, 1, path);
    float* FC = LoadFC(100, 10, path);
    //Uncompleted
}
