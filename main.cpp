#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;

float* LoadDate(char* path) {
    
}

float* LoadConv(const int kernel_size, const int in_channel, const int out_channel, char* path) {
    ifstream weights_file;
    weights_file.open(path, ios::in);
    int weights_size = kernel_size * kernel_size * in_channel * out_channel;
    float* weights = new float[weights_size];
    for( int i = 0; i < weights_size; i++ ) {
        weights_file >> weights[i];
    }
    weights_file.close();
    return weights;
}

void Convolution(const int kernel_size, const int pad, const int stride, const int width, const int height,
         const int in_channel, const int out_channel, const float* weights, const float* inputs, float* outputs ) {
    int out_width = ceil((width + 2 * pad - kernel_size) / stride + 1);
    int out_height = ceil((height + 2 * pad - kernel_size) / stride + 1);
    for( int co = 0; co < out_channel; co++ ) {
        int weights_offset = co * kernel_size * kernel_size * in_channel;
        for( int ho = 0; ho < out_height; ho++ ) {
            for( int wo = 0; wo < out_width; wo++ ) {
                outputs[co * out_width * out_height + ho * out_width + wo] = 0;
                for( int ci = 0; ci < in_channel; ci++ ) {
                    int start_index = ci * height * width + ho * stride * width + wo * stride;
                    for( int kh = 0; kh < kernel_size; kh++) {
                        for( int kw = 0; kw < kernel_size; kw++) {
                            int h_index = ho * stride + kh;
                            int w_index = wo * stride + kw;
                            if( h_index <= pad || w_index <= pad || h_index > height + pad || w_index > width + pad ) {
                                outputs[co * out_width * out_height + ho * out_width + wo] += 0;   
                            } else {
                                outputs[co * out_width * out_height + ho * out_width + wo] += inputs[start_index + kh * width + kw]
                                        * weights[weights_offset + ci * kernel_size * kernel_size + kh * kernel_size * kw]; 
                            }
                        }
                    }
                }
            }
        }   
    } 
}

float* LoadFC(const int in_neuron, const int out_neuron, char* path) {
    ifstream weights_file;
    weights_file.open(path);
    int weights_size = in_neuron * out_neuron;
    float* weights = new float[weights_size];
    for( int i = 0; i < weights_size; i++ ) {
        weights_file >> weights[i];
    }
    weights_file.close();
    return weights;
}

void FullyConnected(const int in_neuron, const int out_neuron, const float* weights, const float* inputs, float* outputs) {
    for( int i = 0; i < out_neuron; i++ ) {
        outputs[i] = 0;
        for( int j = 0; j < in_neuron; j++ ) {
            outputs[i] += inputs[j] * weights[i * in_neuron + j];
        }
    }
}

void ReLU(const float* inputs, const int dim, float* outputs) {
    for( int i = 0; i < dim; i++ ) {
       outputs[i] = (inputs[i] > 0) ? inputs[i] : 0; 
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
    float scale = 0;
    for( int i = 0; i < dim; i++ ) {
        scale +=  exp(inputs[i]);
    } 
    for( int i = 0; i < dim; i++ ) {
        outputs[i] = exp(inputs[i]) / scale;
    }
}

int main() {
    //Uncompleted
    
}
