#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "cltune.h"
#include "packed_array_impl.h"
using namespace std;

int main() {

    // size of image
    const int M1 = 500;
    const int N1 = 500;

    // size of filter
    // must be odd, can be at most group_size*2+1
    const int M2 = 13;
    const int N2 = 13;
    const int group_size = 10;

    PackedArrayImpl imgArr("imgArr", 32, M1*N1);
    for(int i=0; i<M1*N1; i++) {
        imgArr.set(i, rand()%10);
    }

    ifstream infile{ "./conv_prefetch.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string header = imgArr.getConfig().generateOpenCLCode(true, group_size, M2/2);
    string full_kernel = header + kernel;
    cout<<full_kernel<<endl;

    int32_t *imgCells = imgArr.getCells();
    vector<int32_t> img(imgCells, imgCells+imgArr.physical_capacity());
    vector<int32_t> filter(M2*N2);
    vector<int32_t> target(M1*N1);
    vector<int32_t> output(M1*N1);

    for(int i=0; i<M2*N2; i++) {
        filter[i] = rand()%5;
    }

    // column major
    // this loop is super slow lmao -_-
    // eddy while you're optimizing, you might want to cache this
    for(int i=0; i<N1; i++) {
        for(int j=0; j<M1; j++) {
            int fi = 0;
            for(int di=-N2/2; di<=N2/2; di++, fi++) {
                int fj = 0;
                for(int dj=-M2/2; dj<=M2/2; dj++, fj++) {
                    int final_i = i+di;
                    int final_j = j+dj;
                    if( 0 <= final_i && final_i < N1 && 0 <= final_j && final_j < M1) {
                        target[i*M1+j] += filter[fi*M2 + fj] * img[(i+di)*M1 + (j+dj)];
                    }
                }
            }
        }
    }


    cltune::Tuner tuner(size_t{0}, size_t{1});
    auto id = tuner.AddKernelFromString(full_kernel, "conv_prefetch", {M1, N1}, {group_size, group_size});
    tuner.AddKernel({"./conv_simple.cl"}, "conv_simple", {M1, N1}, {group_size, group_size});
    tuner.SetReference({"./conv_ref.cl"}, "ref", {M1, N1}, {group_size, group_size});

    tuner.AddArgumentInput(img);
    tuner.AddArgumentInput(filter);
    tuner.AddArgumentScalar(M2);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(output);

    // Starts the tuner
    tuner.Tune();

    return 0;
}

// =================================================================================================
