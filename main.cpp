#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <ctime>

using namespace std;

float* LoadData(cv::Mat img) {
    int row = img.rows;
    int col = img.cols;
    float* image_array = new float[row * col];
    for( int i = 0; i < row; i++ ) {
        for( int j = 0; j < col; j++ ) {
                image_array[i * col + j] = static_cast<float>(img.at<uchar>(i,j));
                image_array[i * col + j] /= 255;
                image_array[i * col + j] -= 0.1307;
                image_array[i * col + j] /= 0.3081;
        }
    }
    return image_array;
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

float* Convolution(const int kernel_size, const int pad, const int stride, const int width, const int height,
         const int in_channel, const int out_channel, const float* weights, const float* inputs) {
    int out_width = ceil((width + 2 * pad - kernel_size) / stride + 1);
    int out_height = ceil((height + 2 * pad - kernel_size) / stride + 1);
    float* outputs = new float[out_width * out_height * out_channel];
    for( int co = 0; co < out_channel; co++ ) {
        int weights_offset = co * kernel_size * kernel_size * in_channel;
        for( int ho = 0; ho < out_height; ho++ ) {
            for( int wo = 0; wo < out_width; wo++ ) {
                outputs[co * out_width * out_height + ho * out_width + wo] = 0;
                for( int ci = 0; ci < in_channel; ci++ ) {
                    int start_index = ci * height * width + (ho * stride - pad) * width + (wo - pad) * stride;
                    for( int kh = 0; kh < kernel_size; kh++) {
                        for( int kw = 0; kw < kernel_size; kw++) {
                            int h_index = ho * stride + kh;
                            int w_index = wo * stride + kw;
                            if( h_index < pad || w_index < pad || h_index >= height + pad || w_index >= width + pad ) {
                                outputs[co * out_width * out_height + ho * out_width + wo] += 0;   
                            } else {
                                outputs[co * out_width * out_height + ho * out_width + wo] += inputs[start_index + kh * width + kw]
                                        * weights[weights_offset + ci * kernel_size * kernel_size + kh * kernel_size + kw]; 
                            }
                        }
                    }
                }
            }
        }   
    }
    return outputs; 
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

float* FullyConnected(const int in_neuron, const int out_neuron, const float* weights, const float* inputs) {
    float* outputs = new float[out_neuron];
    for( int i = 0; i < out_neuron; i++ ) {
        outputs[i] = 0;
        for( int j = 0; j < in_neuron; j++ ) {
            outputs[i] += inputs[j] * weights[i * in_neuron + j];
        }
    }
    return outputs;
}

float* ReLU(const float* inputs, const int dim) {
    float* outputs = new float[dim];
    for( int i = 0; i < dim; i++ ) {
       outputs[i] = (inputs[i] > 0) ? inputs[i] : 0; 
    }
    return outputs;
}

float* MaxPooling(const float* inputs, const int kernel_size, const int stride, const int channel, const int height, const int width) {
    int out_height = static_cast<int>(ceil(static_cast<float>(height - kernel_size) / stride)) + 1; 
    int out_width = static_cast<int>(ceil(static_cast<float>(width - kernel_size) / stride)) + 1;
    float* outputs = new float[out_height * out_width * channel];
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
                            max = inputs[offset + (h_in + kh) * width + w_in +kw];
                        }
                    }
                }
                outputs[c * out_height * out_width + h * out_width + w] = max;
            }
        }
    }
    return outputs;
}

float* Softmax(const float* inputs, const int dim) {
    float* outputs = new float[dim];
    float scale = 0;
    for( int i = 0; i < dim; i++ ) {
        scale +=  exp(inputs[i]);
    } 
    for( int i = 0; i < dim; i++ ) {
        outputs[i] = exp(inputs[i]) / scale;
    }
    return outputs;
}

int main() {
    clock_t start = clock();
    char image_list[] = "list.txt";
    ifstream listfile(image_list);
    char conv1_file[] = "conv1.weight.txt";
    char conv2_file[] = "conv2.weight.txt";
    char fc1_file[] = "fc1.weight.txt";
    char fc2_file[] = "fc2.weight.txt";
    float* conv1 = LoadConv(5, 1, 10, conv1_file);
    float* conv2 = LoadConv(5, 10, 20, conv2_file);
    float* fc1 = LoadFC(320, 50, fc1_file);
    float* fc2 = LoadFC(50, 10, fc2_file);   
    string infile;
    int label;
    int correct = 0;
    int total = 0;
    while( listfile >> infile >> label ) {
        cv::Mat img = cv::imread(infile, -1);
        float* input = LoadData(img);
        float* output_conv1 = Convolution(5, 0, 1, 28, 28, 1, 10, conv1, input);
        delete[] input;
        float* output_pool1 = MaxPooling(output_conv1, 2, 2, 10, 24, 24);
        delete[] output_conv1;
        float* output_relu1 = ReLU(output_pool1, 12*12*10);
        delete[] output_pool1;
        float* output_conv2 = Convolution(5, 0, 1, 12, 12, 10, 20, conv2, output_relu1);
        delete[] output_relu1;
        float* output_pool2 = MaxPooling(output_conv2, 2, 2, 20, 8, 8);
        delete[] output_conv2;
        float* output_relu2 = ReLU(output_pool2, 4*4*20);
        delete[] output_pool2;
        float* output_fc1 = FullyConnected(320, 50, fc1, output_relu2);
        delete[] output_relu2;
        float* output_relu3 = ReLU(output_fc1, 50);
        delete[] output_fc1;
        float* output_fc2 = FullyConnected(50, 10, fc2, output_relu3);
        delete[] output_relu3;
        float* output = Softmax(output_fc2, 10);
        delete[] output_fc2;
        auto pos = max_element(output, output+10);
        int pred = distance(output, pos);
        delete[] output;
        cout << infile << " label: " << label << " prediction: " << pred << endl;
        if( label == pred ) {
            correct++;
        }
        total++;
    }
    cout << "The eval precision is: " << static_cast<float>(correct) / static_cast<float>(total) << endl;
    clock_t end = clock();
    float total_time=(float)(end-start)/CLOCKS_PER_SEC;
    cout << "Total time: " << total_time << endl;
    cout << "The frame rate is: " << 10000 / total_time << endl;
    return 0;
}
