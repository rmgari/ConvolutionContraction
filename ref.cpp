// Build instructions found at: https://pytorch.org/cppdocs/installing.html#minimal-example

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>

#include "tblis.h"
#include "small/Conv2DLayer.hpp"
#include "test_utils.hpp"

using namespace std;
using namespace torch;
using namespace tblis;

typedef long long ll; 

pair<Tensor, ll> libtorch_convolution(int inc, int outc, Tensor input, Tensor kernel, Tensor b, int s, int p) {
    at::set_num_threads(1);

    int k_size = kernel.sizes()[2];
    // create the convolution layer with all given specifications
    nn::Conv2d layer = nn::Conv2d(nn::Conv2dOptions(inc, outc, k_size).stride(s).padding(p));
    layer->weight = kernel;
    layer->bias = b;
    // perform the convolution
    pair<Tensor, ll> returnVal;

    auto start_libtorch = chrono::steady_clock::now();    
    returnVal.first = layer->forward(input);
    auto end_libtorch = chrono::steady_clock::now();

    returnVal.second = chrono::duration_cast<chrono::nanoseconds>(end_libtorch - start_libtorch).count();
    return returnVal;
}

float max_abs_diff(tblis::tensor<float> t1, tblis::tensor<float> t2, int Co, int Wo, int Ho) {
    float mad = 0.0;
    for (int i = 0; i < Co; ++i) {
        for (int j = 0; j < Wo; ++j) {
            for (int k = 0; k < Ho; ++k) {
                mad = max(mad, abs(t1(i, j, k) - t2(i, j, k)));
            }
        }
    }
    return mad;
}

float max_abs_diff(torch::Tensor t1, tblis::tensor<float> t2, int Co, int Wo, int Ho) {
    float mad = 0.0;
    for (int i = 0; i < Co; ++i) {
        for (int j = 0; j < Wo; ++j) {
            for (int k = 0; k < Ho; ++k) {
                mad = max(mad, abs(t1[0][i][j][k].item<float>() - t2(i, j, k)));
            }
        }
    }
    return mad;
}


int main() {
    // leave uncommented when testing alexnet. comment when testing vgg16. 
    jit::script::Module model = jit::load("alexnet.pt");
    // leave uncommented when testing vgg16. comment when testing alexnet.     
    // jit::script::Module model = jit::load("vgg16.pt");    

    Tensor input = torch::randn({1, 3, 64, 64});
    vector<c10::IValue> model_in;

    vector<Tensor> inputs;
    vector<Tensor> layer_outputs;

    auto params = model.named_parameters();
    auto children = model.named_children();

    // leave uncommented when testing alexnet. comment when testing vgg16. 
    string clayer_inds[5] = {"0", "3", "6", "8", "10"};
    // leave uncommented when testing vgg16. comment when testing alexnet. 
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
        // for vgg16 (leave as is when testing SMaLL, since strides > 2 are unsupported). comment when testing alexnet (in non SMaLL context)
        s = 1, p = 1;
        int Wo = ((Wi - Wf + 2 * p) / s) + 1, Ho = ((Hi - Hf + 2 * p) / s) + 1;

        pair<Tensor, ll> libtorch_out = libtorch_convolution(Ci, Co, inputs[layer], kernels[layer], torch::zeros(Co), s, p);
        Tensor res = libtorch_out.first;
        auto elapsed_libtorch = libtorch_out.second;

        tblis::tensor<float> A_CWH = varray({Ci, Wi + 2 * p, Hi + 2 * p}, 0);
        tblis::tensor<float> A_CHW = varray({Ci, Hi + 2 * p, Wi + 2 * p}, 0);
        tblis::tensor<float> A_HCW = varray({Hi + 2 * p, Ci, Wi + 2 * p}, 0);
        tblis::tensor<float> A_HWC = varray({Hi + 2 * p, Wi + 2 * p, Ci}, 0);
        tblis::tensor<float> A_WHC = varray({Wi + 2 * p, Hi + 2 * p, Ci}, 0);
        tblis::tensor<float> A_WCH = varray({Wi + 2 * p, Ci, Hi + 2 * p}, 0);
        tblis::tensor<float> B = varray({Ci, Co, Wf, Hf}, 0);
        tblis::tensor<float> C = varray({Co, Wo, Ho}, 0);

        // Attempting to incorporate SMaLL
        #if defined(QUANTIZED)
            using BufferT = small::QUInt8Buffer;
        #else
            using BufferT = small::FloatBuffer;
        #endif        

        LayerParams params {Ci, Hi, Wi, Wf, s, small::PADDING_F, Co};
        small::shape_type input_shape{1U, params.C_i, params.H, params.W};

        BufferT filters(params.C_i*params.k*params.k*params.C_o);
        small::init(filters, filters.size());

        small::Conv2DLayer conv2d(input_shape,
                                  params.k, params.k,
                                  params.s, params.p,
                                  params.C_o,
                                  filters,
                                  false); 
        // output for debugging
        cout << Co << ' ' << Wo << ' ' << Ho << ' ' << conv2d.output_shape() << '\n';                                  
        cout << params.C_i << ' ' << params.H << ' ' << params.W  <<  ' ' << params.k << ' ' << params.s << ' ' << params.C_o << '\n';
                                           
        small::Tensor<BufferT> input(input_shape);
        small::Tensor<BufferT> output(conv2d.output_size());

        conv2d.compute_output({&input}, &output);       

        // copy inputs into all As
        for (int i = 0; i < Ci; ++i) {
            for (int j = p; j < Wi + p; ++j) {
                for (int k = p; k < Hi + p; ++k) {
                    A_CWH(i, j, k) = inputs[layer][0][i][j - p][k - p].item<float>();
                    A_CHW(i, k, j) = inputs[layer][0][i][j - p][k - p].item<float>();
                    A_HCW(k, i, j) = inputs[layer][0][i][j - p][k - p].item<float>();
                    A_HWC(k, j, i) = inputs[layer][0][i][j - p][k - p].item<float>();
                    A_WHC(j, k, i) = inputs[layer][0][i][j - p][k - p].item<float>();
                    A_WCH(j, i, k) = inputs[layer][0][i][j - p][k - p].item<float>();
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

        float FLOPS = 2 * Co * Ho * Wo * Hf * Wf * Ci;

        cout << "Layer " << layer + 1 << '\n';
        cout << "Co\tCi\tWf\tHf\tWi\tHi\ts\tp\n";
        cout << Co << '\t' << Ci << '\t' << Wf << '\t' << Hf << '\t' << Wi << '\t' << Hi << '\t' << s << '\t' << p << '\n';
        cout << "LibTorch: " << FLOPS / elapsed_libtorch << " GFLOPs\n";

        // fix kernel, output -> determine correct index mapping in input  
        // second and fourth by s

        tblis::tensor<float> A2_CWH = varray_view<float>({Ci, Wo, Wf, Ho, Hf}, A_CWH.data(), {(Wi + 2 * p) * (Hi + 2 * p), (Hi + 2 * p) * s, (Hi + 2 * p), s, 1});
        tblis::tensor<float> C2_CWH = varray({Co, Wo, Ho}, 0);
        auto start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_CWH, "abcde", B, "afce", 0, C2_CWH, "fbd");
        auto end_tblis = chrono::steady_clock::now();
        auto elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(CWH): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(CWH): " << max_abs_diff(res, C2_CWH, Co, Wo, Ho) << "\n";

        tblis::tensor<float> A2_CHW = varray_view<float>({Ci, Ho, Hf, Wo, Wf}, A_CHW.data(), {(Wi + 2 * p) * (Hi + 2 * p), (Wi + 2 * p) * s, (Wi + 2 * p), s, 1});
        tblis::tensor<float> C2_CHW = varray({Co, Wo, Ho}, 0);
        start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_CHW, "adebc", B, "afce", 0, C2_CHW, "fbd");
        end_tblis = chrono::steady_clock::now();
        elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(CHW): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(CHW): " << max_abs_diff(res, C2_CHW, Co, Wo, Ho) << "\n";

        tblis::tensor<float> A2_HCW = varray_view<float>({Ho, Hf, Ci, Wo, Wf}, A_HCW.data(), {(Wi + 2 * p) * Ci * s, (Wi + 2 * p) * Ci, (Wi + 2 * p), s, 1});
        tblis::tensor<float> C2_HCW = varray({Co, Wo, Ho}, 0);
        start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_HCW, "deabc", B, "afce", 0, C2_HCW, "fbd");
        end_tblis = chrono::steady_clock::now();
        elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(HCW): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(HCW): " << max_abs_diff(res, C2_HCW, Co, Wo, Ho) << "\n";

        tblis::tensor<float> A2_HWC = varray_view<float>({Ho, Hf, Wo, Wf, Ci}, A_HWC.data(), {(Wi + 2 * p) * Ci * s, (Wi + 2 * p) * Ci, Ci * s, Ci, 1});
        tblis::tensor<float> C2_HWC = varray({Co, Wo, Ho}, 0);
        start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_HWC, "debca", B, "afce", 0, C2_HWC, "fbd");
        end_tblis = chrono::steady_clock::now();
        elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(HWC): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(HWC): " << max_abs_diff(res, C2_HWC, Co, Wo, Ho) << "\n";

        tblis::tensor<float> A2_WHC = varray_view<float>({Wo, Wf, Ho, Hf, Ci}, A_WHC.data(), {(Hi + 2 * p) * Ci * s, (Hi + 2 * p) * Ci, Ci * s, Ci, 1});
        tblis::tensor<float> C2_WHC = varray({Co, Wo, Ho}, 0);
        start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_WHC, "bcdea", B, "afce", 0, C2_WHC, "fbd");
        end_tblis = chrono::steady_clock::now();
        elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(WHC): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(WHC): " << max_abs_diff(res, C2_WHC, Co, Wo, Ho) << "\n";

        tblis::tensor<float> A2_WCH = varray_view<float>({Wo, Wf, Ci, Ho, Hf}, A_WCH.data(), {(Hi + 2 * p) * Ci * s, (Hi + 2 * p) * Ci, (Hi + 2 * p), s, 1});
        tblis::tensor<float> C2_WCH = varray({Co, Wo, Ho}, 0);
        start_tblis = chrono::steady_clock::now();        
        mult<float>(1, A2_WCH, "bcade", B, "afce", 0, C2_WCH, "fbd");
        end_tblis = chrono::steady_clock::now();
        elapsed_tblis = chrono::duration_cast<chrono::nanoseconds>(end_tblis - start_tblis).count();
        cout << "TBLIS(WCH): " << FLOPS / elapsed_tblis << " GFLOPs\n"; 
        cout << "Max diff between LibTorch and TBLIS(WCH): " << max_abs_diff(res, C2_WCH, Co, Wo, Ho) << "\n\n";        

        // Source: https://stackoverflow.com/questions/73902752/how-can-i-get-the-maximum-values-of-a-tensor-along-a-dimension
        // cout << "Max diff: " << (torch::max(torch::abs(layer_outputs[layer] - res))).item() << '\n';
        // check max diff between res and C2
        // tblis::tensor<float> C_algo_two = varray({Co, Wo, Ho}, 0);
        // // writing algo 2
        // auto start_algo2 = chrono::steady_clock::now();          
        // for (int l = 0; l < Ho; ++l) {
        //     for (int n = 0; n < Hf; ++n) {
        //         for (int m = 0; m < Wf; ++m) {
        //             for (int i = 0; i < Ci; ++i) {
        //                 for (int k = 0; k < Wo; ++k) {
        //                     for (int j = 0; j < Co; ++j) {
        //                         C_algo_two(j, k, l) += A_CWH(i, k * s + m, l * s + n) * B(i, j, m, n);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        // auto end_algo2 = chrono::steady_clock::now();
        // auto elapsed_algo2 = chrono::duration_cast<chrono::nanoseconds>(end_algo2 - start_algo2).count();              
        // cout << "Algo 2: " << elapsed_algo2 << " nanoseconds\n";        
        // cout << "Max diff between algo 2 and LibTorch: " << max_abs_diff(res, C_algo_two, Co, Wo, Ho) << '\n';
        // cout << "Max diff between algo 2 and TBLIS: " << max_abs_diff(C_algo_two, C2, Co, Wo, Ho) << '\n';
    }
    
    return 0;
}