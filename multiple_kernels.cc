#include <vector>
#include <chrono>
#include <random>
#include <iostream>

// Includes the OpenCL tuner library
#include "cltune.h"
#include "packed_array_impl.h"
using namespace std;

int main() {
    PackedArrayImpl a("packed_input", 2, 1024 * 1024 * 64);

    // Sets the filenames of the OpenCL kernels
    auto simpleSum = std::vector<std::string>{"./sum.cl"};
    auto packedSum = std::vector<std::string>{"./packed_sum.cl.run"};

    // Matrix size
    const auto simpleSize = size_t{128*1024*1024};
    const auto packedSize = size_t{128*1024*1024};

    // Creates data structures
    std::vector<int32_t> arr(simpleSize);
    std::vector<int32_t> sum(1);

    // Initializes the tuner (platform 0, device 0)
    cltune::Tuner tuner(size_t{0}, size_t{0});

    // Adds a kernel which supports unrolling through the UNROLL parameter. Note that the kernel
    // itself needs to implement the UNROLL parameter and (in this case) only accepts a limited
    // amount of values.
    auto id = tuner.AddKernel(simpleSum, "sum", {simpleSize}, {1});
    id = tuner.AddKernel(packedSum, "packed_sum", {packedSize}, {1});

    // Sets the function's arguments. Note that all kernels have to accept (but not necessarily use)
    // all input arguments.
    tuner.AddArgumentInput(arr);
    tuner.AddArgumentOutput(sum);

    // Starts the tuner
    tuner.Tune();

    // Prints the results to screen
    tuner.PrintToScreen();

    return 0;
}

// =================================================================================================
