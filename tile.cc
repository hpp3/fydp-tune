#include <vector>
#include <iostream>
#include <fstream>

#include "cltune.h"
#include "packed_array_impl.h"
using namespace std;

int main() {

    uint32_t M = 2;
    uint32_t N = 2;
    uint32_t K = 3;
    uint32_t group_size = 1;

    PackedArrayImpl a("arrA", 8, M*K);
    PackedArrayImpl b("arrB", 8, K*N);

    for(int i=0; i<M*K; i++) {
        a.set(i, i+1);
    }

    for(int i=0; i<K*N; i++) {
        b.set(i, i+1);
    }


    ifstream infile{ "./multi_tile.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string headerA = a.getConfig().generateOpenCLCode(true, group_size);
    string headerB = b.getConfig().generateOpenCLCode(true, group_size);
    string full_kernel = headerA + headerB + kernel;

    cout<<full_kernel<<endl;

    int32_t *cellA = a.getCells();
    int32_t *cellB = b.getCells();
    std::vector<int32_t> arrA(cellA, cellA+a.physical_capacity());
    std::vector<int32_t> arrB(cellB, cellB+b.physical_capacity());
    std::vector<int32_t> target(M*N);
    std::vector<int32_t> arrC(M*N);

    target[0] = 22;
    target[1] = 28;
    target[2] = 49;
    target[3] = 64;

    cltune::Tuner tuner(size_t{0}, size_t{0});
    auto id = tuner.AddKernelFromString(full_kernel, "multi_tile", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(arrA);
    tuner.AddArgumentInput(arrB);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);


    // Starts the tuner
    tuner.Tune();

    return 0;
}

// =================================================================================================
