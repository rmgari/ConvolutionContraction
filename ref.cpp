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
    
    // for (int i = 0; i < kernels.size(); ++i) {
    //     int outc = kernels[i].sizes()[0];
    //     int inc = kernels[i].sizes()[1];

    //     Tensor res = naive_algo(inc, outc, inputs[i], kernels[i], biases[i], specs[i].first, specs[i].second);
    //     // Tensor res = naive_algo(inc, outc, inputs[i], kernels[i], biases[i], 1, 1);

    //     // Source: https://stackoverflow.com/questions/73902752/how-can-i-get-the-maximum-values-of-a-tensor-along-a-dimension
    //     cout << "Max diff: " << (torch::max(torch::abs(layer_outputs[i] - res))).item() << '\n';
    // }

    // cout << '\n';

    // setup algo 2
    int Wi=5, Hi=5, Ci=1;
    int Wf=3, Hf=3;
    // assume stride = 1, padding = 0
    int Wo = Wi - Wf + 1, Ho = Wi - Wf + 1, Co = 1;

    Tensor a2_input = torch::empty({Ci, Wi, Hi});
    int fill = 1;
    for (int i = 0; i < Wi; ++i) {
        for (int j = 0; j < Hi; ++j) {
            a2_input[0][i][j] = fill++;
        }
    }
    
    Tensor a2_kernel = torch::empty({Co, Ci, Wf, Hf});
    fill = 1;
    for (int i = 0; i < Wf; ++i) {
        for (int j = 0; j < Hf; ++j) {
            a2_kernel[0][0][i][j] = fill++;
        }
    }

    Tensor c_ref = naive_algo(Ci, Co, a2_input, a2_kernel, torch::zeros(Co), 1, 0);
    
    // vector<len_type> dimsA = {Ci, Wi, Hi};
    // tblis::tensor<float> A(dimsA, COLUMN_MAJOR); 
    tblis::tensor<float> A = varray({Ci, Wi, Hi}, 0);    
    tblis::tensor<float> B = varray({Ci, Co, Wf, Hf}, 0);
    tblis::tensor<float> C = varray({Co, Wo, Ho}, 0);

    // copy over A, B
    for (int i = 0; i < Ci; ++i) {
        for (int j = 0; j < Wi; ++j) {
            for (int k = 0; k < Hi; ++k) {
                A(i, j, k) = a2_input[i][j][k].item<float>();
            }
        }
    }

    for (int i = 0; i < Co; ++i) {
        for (int j = 0; j < Ci; ++j) {
            for (int k = 0; k < Wf; ++k) {
                for (int l = 0; l < Hf; ++l) {
                    B(j, i, k, l) = a2_kernel[i][j][k][l].item<float>();
                }
            }
        }
    }    

    // writing algo 2
    for (int l = 0; l < Ho; ++l) {
        for (int n = 0; n < Hf; ++n) {
            for (int m = 0; m < Wf; ++m) {
                for (int i = 0; i < Ci; ++i) {
                    for (int k = 0; k < Wo; ++k) {
                        for (int j = 0; j < Co; ++j) {
                            C(j, k, l) += A(i, k + m, l + n) * B(i, j, m, n);
                        }
                    }
                }
            }
        }
    }

    // testing algo 2
    cout << A << '\n';
    cout << B << '\n';
    cout << c_ref << '\n';
    cout << C << '\n';

    // fix kernel, output -> determine correct index mapping in input    
    
    tblis::tensor<float> A2 = varray_view<float>({Ci, Wo, Wf, Ho, Hf}, A.data(), {Wi * Hi, Hi, Hi, 1, 1});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cout << A2(0, 0, i, 0, j) << ' ';
        }
        cout << '\n';
    }
    tblis::tensor<float> C2 = varray({Co, Wo, Ho}, 0);
    mult<float>(1, A2, "abcde", B, "afce", 0, C2, "fbd");
    cout << C2 << '\n';    
    return 0;
}