#include <vector>
#include <iostream>
#include <fstream>

#include "cltune.h"
#include "packed_array_impl.h"
#include "packed_2d_array_impl.h"
using namespace std;

int main() {

    uint32_t M = 2;
    uint32_t N = 2;
    uint32_t K = 3;
    uint32_t group_size = 1;
    int bitsize = 32;

    Packed2DArrayImpl a("arrA", bitsize, M, K, true);  // Row major
    Packed2DArrayImpl b("arrB", bitsize, K, N, false); // Column major

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a.set(i, j, i*j);
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            b.set(i, j, i*j);
        }
    }

    ifstream infile{ "./multi.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    // Note: generateOpenCLCode is called on a and b NOT on their config
    string headerA = a.generateOpenCLCode(false, group_size); // no prefetch
    string headerB = b.generateOpenCLCode(false, group_size); // no prefetch
    string full_kernel = headerA + headerB + kernel;

    cout<<full_kernel<<endl;

    int32_t *cellA = a.getCells();
    int32_t *cellB = b.getCells();
    std::vector<int32_t> arrA(cellA, cellA+a.physical_capacity());
    std::vector<int32_t> arrB(cellB, cellB+b.physical_capacity());
    std::vector<int32_t> target(M*N);
    std::vector<int32_t> arrC(M*N);

    // Need to fix this part to expected result
    target[0] = 22;
    target[1] = 28;
    target[2] = 49;
    target[3] = 64;

    cltune::Tuner tuner(size_t{0}, size_t{0});
    auto id = tuner.AddKernelFromString(full_kernel, "multi", {M, N}, {group_size, group_size});
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
