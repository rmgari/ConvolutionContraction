// Build instructions found at: https://pytorch.org/cppdocs/installing.html#minimal-example

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include "tblis.h"

using namespace std;
using namespace torch;
using namespace tblis;

Tensor naive_algo(int inc, int outc, Tensor input, Tensor kernel, Tensor b, int s, int p) {
    int k_size = kernel.sizes()[2];
    // create the convolution layer with all given specifications
    nn::Conv2d layer = nn::Conv2d(nn::Conv2dOptions(inc, outc, k_size).stride(s).padding(p));
    layer->weight = kernel;
    layer->bias = b;
    // perform the convolution
    return layer->forward(input);
}

int main() {
    

    jit::script::Module model = jit::load("alexnet.pt");

    Tensor input = torch::randn({1, 3, 64, 64});
    vector<c10::IValue> model_in;

    // TODO: copy input tensor into A, use tblis view
    // for test case one (pass through first layer of AlexNet)
    tblis::tensor<float> A = varray({3, 15, 15, 11, 11}, 0);
    tblis::tensor<float> B = varray({64, 3, 11, 11}, 0);

    vector<Tensor> inputs;
    vector<Tensor> layer_outputs;

    auto params = model.named_parameters();
    auto children = model.named_children();

    string clayer_inds[5] = {"0", "3", "6", "8", "10"};
    // string clayer_inds[13] = {"0", "2", "5", "7", "10", "12", "14", "17", "19", "21", "24", "26", "28"};
    
    model_in.push_back(input);

    for (auto c : children) {
        if (c.name == "features") {
            auto fchild = c.value.named_children();
            for (auto x : fchild) {
                if (find(begin(clayer_inds), end(clayer_inds), x.name) != end(clayer_inds)) {
                    inputs.push_back(model_in[0].toTensor().clone());
                    auto out = x.value.forward(model_in); 
                    layer_outputs.push_back(out.toTensor().clone());
                    model_in[0] = out;                                      
                } else {
                    auto out = x.value.forward(model_in);
                    model_in[0] = out;
                }
            }
        }
    }

    vector<Tensor> kernels;
    vector<Tensor> biases;    

    for (auto p : params) {
        // extracting the kernels from the pre-trained model
        if (p.name.find("features") != string::npos && p.name.find("weight") != string::npos) {
            kernels.push_back(p.value.clone());
        }
        // extracting the biases from the pre-trained model
        if (p.name.find("features") != string::npos && p.name.find("bias") != string::npos) {
            biases.push_back(p.value.clone());
        }
    }

    pair<int, int> specs[5];

    // stride, padding pairs for each convolution layer of AlexNet
    specs[0] = make_pair(4, 2);
    specs[1] = make_pair(1, 2);
    specs[2] = make_pair(1, 1);
    specs[3] = make_pair(1, 1);
    specs[4] = make_pair(1, 1);    
    
    for (int i = 0; i < kernels.size(); ++i) {
        int outc = kernels[i].sizes()[0];
        int inc = kernels[i].sizes()[1];

        Tensor res = naive_algo(inc, outc, inputs[i], kernels[i], biases[i], specs[i].first, specs[i].second);
        // Tensor res = naive_algo(inc, outc, inputs[i], kernels[i], biases[i], 1, 1);

        // Source: https://stackoverflow.com/questions/73902752/how-can-i-get-the-maximum-values-of-a-tensor-along-a-dimension
        cout << "Max diff: " << (torch::max(torch::abs(layer_outputs[i] - res))).item() << '\n';
    }

    for (int i = 0; i < 63; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 11; ++k)
                for (int l = 0; l < 11; ++l)
                    B(i, j, k, l) = kernels[0][i][j][k][l].item<float>();
    
    tblis::tensor<float> C = varray({15, 15, 64}, 0);
    
    // mult<float>(1, A, "abcde", B, "fade", 0, C, "bcf");

    return 0;
}

// Algorithm outline for performing and checking the contractions

// For each layer of a given model
// Copy the input and kernel Tensors from pytorch into TBLIS tensors
// Copy the output Tensor from pytorch into a TBLIS tensor as well

// Is the overhead added from this step something we should account for and try to reduce (approaching im2col)? Given how I've written the contraction, the operation is dependent on the input tensor being expanded into a tensor with higher dimensionality before the contraction happens. Is there any way to avoid this (maybe by rewriting the contraction)?

// Initialize output tensor C with the right dimensions (can do this in a function using the formula [(W−K+2P)/S]+1)
// Call mult<T>(alpha, A, idx_A, B, idx_B, beta, C, idx_C), where alpha = 1, beta = 1, A = input tensor, B = kernel, C = output, idx_A = "", idx_B = "", idx_C = ""

// Call add<T>(alpha, A, idx_A, beta, B, idx_B) with alpha = 1, beta = -1, A = output_ref, B = C, idx_A = all indices of output_ref, idx_B = all indices of C
// On the result tensor, we can call reduce<T>(op, A, idx_A), where op = REDUCE_MAX_ABS == REDUCE_NORM_INF, idx_A = all indices of the result tensor to find the max diff

// Does the extra addition step introduce alter the acceptable bounds for the max-diff?
