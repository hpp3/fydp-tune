#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "cltune.h"
#include "cpack.h"
#include "packed_2d_array_impl.h"
#include "packed_array_impl.h"

using namespace std;
//filter <= 2 gr + 1

void easy_reduce() {
    TUNE_START(vector<int>({2, 4, 8, 16, 32}), vector<int>({1,2,4,8,16,32}), vector<bool>({false}));
    //TUNE_START(vector<int>({2, 8, 32}), vector<int>({1,8,32}), vector<bool>({false}));
    const auto simpleSize = int32_t{1 << 26};

    std::vector<int32_t> data;
    std::vector<int32_t> sum(1);

    for (size_t i = 0; i < simpleSize; i++) {
        data.push_back(rand() % 4);
    }
    PROCESS_VEC(data, "test");

    std::string kernel = KERNEL_STRING("./reduce.opencl");

    float total_time = 0;
    for (int i = 0; i < 100; i++) {
        cltune::Tuner tuner(size_t{0}, size_t{0});

        auto id = tuner.AddKernelFromString(kernel, "reduction_sum_scalar_complete2", {(unsigned long)simpleSize}, {(unsigned long)cpack_groupsize});

        auto start = chrono::steady_clock::now();
        tuner.AddArgumentInput(data);
        auto time = chrono::steady_clock::now() - start;
        const auto timing = std::chrono::duration<float,std::milli>(time).count();
        total_time += timing;
    }
    cpack_times.push_back(std::make_pair(std::make_tuple(cpack_bitsize, cpack_groupsize, cpack_prefetch), total_time)); \
}}}
    std::sort(cpack_times.begin(), cpack_times.end(), [](const pair<tuple<int, int, bool>, double>& a, const pair<tuple<int, int, bool>, double>& b) -> bool {  \
        return a.second < b.second;                                                                                         \
    });                                                                                                                     \
    std::cout << "Summary:\nBitsize\tGrpSize\tPrefch\tTime (ms)" << std::endl;                                              \
    for (size_t i = 0; i < cpack_times.size(); i++) {                                                                       \
        std::cout << std::get<0>(cpack_times[i].first) << "\t" << std::get<1>(cpack_times[i].first) << "\t" << std::get<2>(cpack_times[i].first) << "\t" << cpack_times[i].second << std::endl; \
    }

    //tuner.AddArgumentInput(sum);

    //tuner.SetNumRuns(100);
    //tuner.Tune();
    //TUNE_END(tuner);
}

void tile() {
    uint32_t dim = 2;
    uint32_t M = dim; 
    uint32_t N = dim;
    uint32_t K = 2;
    uint32_t group_size = 1;

    PackedArrayImpl a("arrA", 32, M*K);
    PackedArrayImpl b("arrB", 32, K*N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.set(i*M + j, (i*M + j));
        }
    }
    
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.set(i*M + j, (20 + i*K + j));
        }
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

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            target[i*N + j] = 0;
            for (size_t k = 0; k < K; k++) {
                target[i*N+ j] += a.get(i*K+k) * b.get(k*N+j); 
                //cout << "a[" << i << "][" << k << "] = " << a.get(i*K+k);
                //cout << " * b[" << k << "][" << j << "] = " << b.get(k*N+j) << endl;
            }
        }
    }

    for (auto x : target) cout << x << " ";
    cout << endl;
    cout << endl;
    for (auto x : arrA) cout << x << " ";
    cout << endl;
    for (auto x : arrB) cout << x << " ";
    cout << endl;

    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(full_kernel, "multi_tile", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

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
    TUNE_START(vector<int>({32}), vector<int>({8,16,32}), vector<bool>({true, false}));
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

    std::string kernel = KERNEL_STRING("./multi_tile.cl");
    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(kernel, "multi_tile", {M, N}, {cpack_groupsize, cpack_groupsize});
    //tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {cpack_groupsize, cpack_groupsize});

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

void easy_mult() {
    TUNE_START(vector<int>({32}), vector<int>({1,2,4,8,16,32}), vector<bool>({false}));
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

    std::string kernel = KERNEL_STRING("./matrix.cl");
    cltune::Tuner tuner(size_t{0}, size_t{0});
    tuner.AddKernelFromString(kernel, "matrixMult", {M, N}, {cpack_groupsize, cpack_groupsize});
    //tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {cpack_groupsize, cpack_groupsize});

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


    ifstream infile{ "./matrix.cl" };
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
     tuner.AddKernelFromString(full_kernel, "matrixMult", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

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

void rowcol() {
    uint32_t dim = 10;
    uint32_t M = dim;
    uint32_t N = dim;
    uint32_t K = dim;
    uint32_t group_size = 1;
    int bitsize = 32;

    Packed2DArrayImpl a("arrA", bitsize, M, K, false);  // Row major
    Packed2DArrayImpl b("arrB", bitsize, K, N, false); // Column major

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.set(i, j, j+i*5);
        }
    }

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.set(i, j, j+i*3);
        }
    }

    ifstream infile{ "./new_multi.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    // Note: generateOpenCLCode is called on a and b NOT on their config
    string headerA = a.generateOpenCLCode(true, group_size); // no prefetch
    string headerB = b.generateOpenCLCode(true, group_size); // no prefetch
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
                target[i*N+ j] += a.get(i,k) * b.get(k, j); 
                //cout << "a[" << i << "][" << k << "] = " << a.get(i, k) << endl;
                //cout << "b[" << k << "][" << j << "] = " << b.get(k, j) << endl;
            }
            //cout << target[i*N + j] << " ";
        }
        //cout << endl;
    }

    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(full_kernel, "multi", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(arrA);
    tuner.AddArgumentInput(arrB);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(10);


    // Starts the tuner
    tuner.Tune();
}

void both() {
    uint32_t dim = 1024;
    uint32_t M = dim;
    uint32_t N = dim;
    uint32_t K = dim;
    uint32_t group_size = 32;

    std::vector<int32_t> arrA;
    std::vector<int32_t> arrAt;
    std::vector<int32_t> arrB;
    std::vector<int32_t> arrBt;
    std::vector<int32_t> target(M*N);
    std::vector<int32_t> arrC(M*N);

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            arrA.push_back((i*M + j) % 16);
        }
    }

        for (size_t j = 0; j < K; j++) {
    for (size_t i = 0; i < M; i++) {
            arrAt.push_back((i*M + j) % 16);
        }
    }

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            arrB.push_back((i*K + j) % 16);
        }
    }

        for (size_t j = 0; j < N; j++) {
    for (size_t i = 0; i < K; i++) {
            arrBt.push_back((i*K + j) % 16);
        }
    }

    ifstream infile{ "./new_multi.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };


    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            target[i*N + j] = 0;
            for (size_t k = 0; k < K; k++) {
                target[i*N+ j] += arrA[i*M+k] * arrB[k*N+j]; 
            }
//            cout << target[i*N + j] << " ";
        }
//        cout << endl;
    }


    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(kernel, "myGEMM2", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(arrA);
    tuner.AddArgumentInput(arrBt);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(10);

    // Starts the tuner
    tuner.Tune();

}

void both_tile() {
    uint32_t dim = 4;
    uint32_t M = dim;
    uint32_t N = dim;
    uint32_t K = 8;
    uint32_t group_size = 2;
    int bitsize = 32;

    Packed2DArrayImpl a("arrA", bitsize, M, K, true);
    Packed2DArrayImpl b("arrB", bitsize, K, N, true);

    /*
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.set(i, j, (i*M + j) % 16);
        }
    }
    
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.set(i, j, (i*K + j) % 16);
        }
    }
    */

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            a.set(i, j, j+i*5);
        }
    }

    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < N; j++) {
            b.set(i, j, j+i*3);
        }
    }

    ifstream infile{ "./multi_tile.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    // Note: generateOpenCLCode is called on a and b NOT on their config
    string headerA = a.generateOpenCLCode(true, group_size); // no prefetch
    string headerB = b.generateOpenCLCode(true, group_size); // no prefetch
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
            target[j*N + i] = 0;
            for (size_t k = 0; k < K; k++) {
                target[j*N+i] += a.get(i,k) * b.get(k,j); 
                //cout << "a[" << i << "][" << k << "] = " << a.get(i, k);
                //cout << " * b[" << k << "][" << j << "] = " << b.get(k, j) << endl;
            }
        }
    }
/*
    for (auto x : target) cout << x << " ";
    cout << endl;
    for (auto x : arrA) cout << x << " ";
    cout << endl;
    for (auto x : arrB) cout << x << " ";
    cout << endl;
    */

    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(full_kernel, "multi_tile", {M, N}, {group_size, group_size});
    tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {group_size, group_size});

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

void easy_both() {
    TUNE_START_2D(vector<int>({32}), vector<int>({8,16,32}), vector<bool>({true, false}), vector<bool>({true, false}));
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

    PROCESS_VEC_2D(a, "arrA", M);
    PROCESS_VEC_2D(b, "arrB", K);

    std::string kernel = KERNEL_STRING_2D("./multi_tile.cl");
    cltune::Tuner tuner(size_t{0}, size_t{0});
     tuner.AddKernelFromString(kernel, "multi_tile", {M, N}, {cpack_groupsize, cpack_groupsize});
    //tuner.SetReference({"./ref.cl"}, "ref", {M, N}, {cpack_groupsize, cpack_groupsize});

    tuner.AddArgumentScalar((int)M);
    tuner.AddArgumentScalar((int)N);
    tuner.AddArgumentScalar((int)K);
    tuner.AddArgumentInput(a);
    tuner.AddArgumentInput(b);
    tuner.AddArgumentInput(target);
    tuner.AddArgumentOutput(arrC);
    tuner.SetNumRuns(100);

    tuner.Tune();
    TUNE_END_2D(tuner);
}
int main() {

    //easy_tile();
    //tile();
    //easy_reduce();
    //rowcol();
    //easy_mult();
    //both_tile();
    //easy_reduce();
    easy_both();
    return 0;
}
