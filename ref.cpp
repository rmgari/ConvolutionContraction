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
    // layer->bias = b;
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
    
    for (int layer = 0; layer < kernels.size(); ++layer) {
        int Co = kernels[layer].sizes()[0];
        int Ci = kernels[layer].sizes()[1];
        int Wi = inputs[layer].sizes()[2], Hi = inputs[layer].sizes()[3];
        int Wf = kernels[layer].sizes()[2], Hf = kernels[layer].sizes()[3];
        int s = specs[layer].first, p = specs[layer].second;
        int Wo = ((Wi - Wf + 2 * p) / s) + 1, Ho = ((Hi - Hf + 2 * p) / s) + 1;

        Tensor res = naive_algo(Ci, Co, inputs[layer], kernels[layer], biases[layer], s, p);      

        tblis::tensor<float> A = varray({Ci, Wi + 2 * p, Hi + 2 * p}, 0);
        tblis::tensor<float> B = varray({Ci, Co, Wf, Hf}, 0);
        tblis::tensor<float> C = varray({Co, Wo, Ho}, 0);

        // copy inputs into A
        for (int i = 0; i < Ci; ++i) {
            for (int j = p; j < Wi + p; ++j) {
                for (int k = p; k < Hi + p; ++k) {
                    A(i, j, k) = inputs[layer][0][i][j - p][k - p].item<float>();
                }
            }
        }       

        // copy kernel into B
        for (int i = 0; i < Co; ++i) {
            for (int j = 0; j < Ci; ++j) {
                for (int k = 0; k < Wf; ++k) {
                    for (int l = 0; l < Hf; ++l) {
                        B(j, i, k, l) = kernels[layer][i][j][k][l].item<float>();
                    }
                }
            }
        }
        // fix kernel, output -> determine correct index mapping in input  
        // second and fourth by s
        tblis::tensor<float> A2 = varray_view<float>({Ci, Wo, Wf, Ho, Hf}, A.data(), {(Wi + 2 * p) * (Hi + 2 * p), s * (Hi + 2 * p), (Hi + 2 * p), s, 1});
        tblis::tensor<float> C2 = varray({Co, Wo, Ho}, 0);
        mult<float>(1, A2, "abcde", B, "afce", 0, C2, "fbd");

        // Source: https://stackoverflow.com/questions/73902752/how-can-i-get-the-maximum-values-of-a-tensor-along-a-dimension
        // cout << "Max diff: " << (torch::max(torch::abs(layer_outputs[layer] - res))).item() << '\n';
        // check max diff between res and C2

        float max_abs_diff = 0;
        for (int i = 0; i < Co; ++i) {
            for (int j = 0; j < Wo; ++j) {
                for (int k = 0; k < Ho; ++k) {
                    max_abs_diff = max(max_abs_diff, abs(res[0][i][j][k].item<float>() - C2(i, j, k)));
                }
            }
        }
        cout << "Co\tCi\tWf\tHf\tWi\tHi\ts\tp\n";
        cout << Co << '\t' << Ci << '\t' << Wf << '\t' << Hf << '\t' << Wi << '\t' << Hi << '\t' << s << '\t' << p << '\n';
        cout << "Max diff: " << max_abs_diff << "\n\n";
    }
    
    return 0;
}

    // Tensor c_ref = naive_algo(Ci, Co, a2_input, a2_kernel, torch::zeros(Co), s, p);
    // // writing algo 2
    // for (int l = 0; l < Ho; ++l) {
    //     for (int n = 0; n < Hf; ++n) {
    //         for (int m = 0; m < Wf; ++m) {
    //             for (int i = 0; i < Ci; ++i) {
    //                 for (int k = 0; k < Wo; ++k) {
    //                     for (int j = 0; j < Co; ++j) {
    //                         C(j, k, l) += A(i, k + m, l + n) * B(i, j, m, n);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }