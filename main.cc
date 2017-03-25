#include <vector>
#include <iostream>
#include <fstream>

#include "cltune.h"
#include "cpack.h"
#include "packed_array_impl.h"
using namespace std;

void tile() {
    uint32_t dim = 4;
    uint32_t M = dim;
    uint32_t N = dim;
    uint32_t K = dim;
    uint32_t group_size = 2;

    PackedArrayImpl a("arrA", 4, M*K);
    PackedArrayImpl b("arrB", 4, K*N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.set(i*M + j, (i*M + j) % 16);
        }
    }
    
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.set(i*M + j, (i*K + j) % 16);
        }
    }

    ifstream infile{ "../samples/fydp-tune/multi_tile.cl" };
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

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            target[i*N + j] = 0;
            for (size_t k = 0; k < K; k++) {
                target[i*N+ j] += a.get(i*K+k) * b.get(k*N+j); 
            }
            cout << target[i*N + j] << " ";
        }
        cout << endl;
    }

    cltune::Tuner tuner(size_t{0}, size_t{0});
    auto id = tuner.AddKernelFromString(full_kernel, "multi_tile", {M, N}, {group_size, group_size});
    tuner.SetReference({"../samples/fydp-tune/ref.cl"}, "ref", {M, N}, {group_size, group_size});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(arrA);
    tuner.AddArgumentInput(arrB);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(1);


    // Starts the tuner
    tuner.Tune();

}

void easy_tile() {
    TUNE_START(vector<int>({4, 8, 16, 32}), vector<int>({8,16,32}), vector<bool>({true, false}));
    //TUNE_START(vector<int>({4}), vector<int>({8}), vector<bool>({true}));
    uint32_t dim = 1024;
    uint32_t M = dim;
    uint32_t N = dim;
    uint32_t K = dim;


    vector<int32_t> a;
    vector<int32_t> b;
    std::vector<int32_t> target(M*N);
    std::vector<int32_t> arrC(M*N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.push_back((i*M + j) % 16);
        }
    }
    
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.push_back((i*K + j) % 16);
        }
    }

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            target[i*N + j] = 0;
            for (size_t k = 0; k < K; k++) {
                target[i*N+ j] += a[i*K+k] * b[k*N+j]; 
            }
        }
    }

    PROCESS_VEC(a, "arrA");
    PROCESS_VEC(b, "arrB");

    std::string kernel = KERNEL_STRING("../samples/fydp-tune/multi_tile.cl");
    cltune::Tuner tuner(size_t{0}, size_t{0});
    auto id = tuner.AddKernelFromString(kernel, "multi_tile", {M, N}, {cpack_groupsize, cpack_groupsize});
    //tuner.SetReference({"../samples/fydp-tune/ref.cl"}, "ref", {M, N}, {cpack_groupsize, cpack_groupsize});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(a);
    tuner.AddArgumentInput(b);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(100);

    tuner.Tune();
    TUNE_END(tuner);
}

void mult() {
    uint32_t gg = 1024;
    uint32_t M = gg;
    uint32_t N = gg;
    uint32_t K = gg;
    uint32_t group_size = 32;

    PackedArrayImpl a("arrA", 32, M*K);
    PackedArrayImpl b("arrB", 32, K*N);


    ifstream infile{ "../samples/fydp-tune/matrix.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string headerA = a.getConfig().generateOpenCLCode(false, group_size);
    string headerB = b.getConfig().generateOpenCLCode(false, group_size);
    string full_kernel = headerA + headerB + kernel;

    cout<<full_kernel<<endl;

    int32_t *cellA = a.getCells();
    int32_t *cellB = b.getCells();
    std::vector<int32_t> arrA(cellA, cellA+a.physical_capacity());
    std::vector<int32_t> arrB(cellB, cellB+b.physical_capacity());
    std::vector<int32_t> target(M*N);
    std::vector<int32_t> arrC(M*N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            target[i*N + j] = 0;
            for (size_t k = 0; k < K; k++) {
                target[i*N+ j] += a.get(i*K+k) * b.get(k*N+j); 
            }
        }
    }

    cltune::Tuner tuner(size_t{0}, size_t{0});
    auto id = tuner.AddKernelFromString(full_kernel, "matrixMult", {M, N}, {group_size, group_size});
    tuner.SetReference({"../samples/fydp-tune/ref.cl"}, "ref", {M, N}, {group_size, group_size});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(arrA);
    tuner.AddArgumentInput(arrB);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(100);


    // Starts the tuner
    tuner.Tune();

}

void easy_reduce() {
    //TUNE_START(vector<int>({2, 4, 8, 16, 32}));
    TUNE_START(vector<int>({2, 4, 8, 16, 32}), vector<int>({8,16,32}), vector<bool>({false}));
    const auto simpleSize = int32_t{1 << 26};

    std::vector<int32_t> data;
    std::vector<int32_t> sum(1);

    for (size_t i = 0; i < simpleSize; i++) {
        data.push_back(rand() % 4);
    }
    PROCESS_VEC(data, "test");

    std::string kernel = KERNEL_STRING("../samples/fydp-tune/reduce.opencl");

    cltune::Tuner tuner(size_t{0}, size_t{0});

    auto id = tuner.AddKernelFromString(kernel, "reduction_sum_scalar_complete2", {(unsigned long)simpleSize}, {(unsigned long)cpack_groupsize});

    tuner.AddArgumentInput(data);
    tuner.AddArgumentInput(sum);

    tuner.SetNumRuns(100);
    tuner.Tune();
    TUNE_END(tuner);
}

int main() {

    //easy_tile();
    //tile();
    easy_reduce();
    return 0;
}
