#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>

#include "cltune.h"
#include "packed_array_impl.h"
#include "json11.hpp"
using namespace std;

void packed_sum(int bitsize) {
    const auto simpleSize = int32_t{1024};
    PackedArrayImpl a("test", bitsize, simpleSize);

    for(int i=0; i<10; i++) {
        a.set(i, 3);
    }

    int group_size = 32;

    cout<<"size: "<<a.size()<<endl;
    cout<<"physical size: "<<a.physical_size()<<endl;
    cout<<"capacity: "<<a.capacity()<<endl;
    cout<<"physical capacity: "<<a.physical_capacity()<<endl;

    // read the file
    ifstream infile{ "../samples/fydp-tune/packed_sum.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string header = a.getConfig().generateOpenCLCode(false, group_size);
    string full_kernel = header + kernel;

    cout<<full_kernel<<endl;


    // Creates data structures
    int32_t *cells = a.getCells();
    std::vector<int32_t> arr(cells, cells+a.physical_capacity());
    std::vector<int32_t> sum(1);
    sum[0] = 666;


    // Initializes the tuner (platform 0, device 0)
    cltune::Tuner tuner(size_t{0}, size_t{0});

    // Adds a kernel which supports unrolling through the UNROLL parameter. Note that the kernel
    // itself needs to implement the UNROLL parameter and (in this case) only accepts a limited
    // amount of values.
    auto id = tuner.AddKernelFromString(full_kernel, "packed_sum", {(unsigned long)a.physical_capacity()}, {(unsigned long)group_size});

    // Sets the function's arguments. Note that all kernels have to accept (but not necessarily use)
    // all input arguments.
    tuner.AddArgumentInput(arr);
    tuner.AddArgumentOutput(sum);

    // Starts the tuner
    tuner.Tune();

    // Prints the results to screen
    tuner.PrintToScreen();
    vector<pair<string,string>> descriptions;
    descriptions.push_back(make_pair("packingParam", "2"));

    tuner.PrintJSON("fydp-tune.json", descriptions);
}

double reduce(int bitsize) {
    int group_size = 32;
    const auto simpleSize = int32_t{1 << 26};
    PackedArrayImpl a("test", bitsize, simpleSize);
    for (size_t i = 0; i < simpleSize; i++) {
        a.set(i, rand() % 4);
    }
    cout<<"bitsize:"<<bitsize<<endl;
    cout<<"size: "<<a.size()<<endl;
    cout<<"physical size: "<<a.physical_size()<<endl;
    cout<<"capacity: "<<a.capacity()<<endl;
    cout<<"physical capacity: "<<a.physical_capacity()<<endl;

    // read the file
    ifstream infile{ "../samples/fydp-tune/reduce.opencl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string header = a.getConfig().generateOpenCLCode(false, group_size);
    string full_kernel = header + kernel;

    //cout<<full_kernel<<endl;

    const int numWorkgroups = (simpleSize + group_size - 1) / group_size;
    //const int localBytes = workgroupSize * sizeof(int);
    
    // Creates data structures
    int32_t *cells = a.getCells();

    std::vector<int32_t> local(numWorkgroups, 0);
    std::vector<int32_t> results(numWorkgroups, 0);
    std::vector<int32_t> data(cells, cells+a.physical_capacity());
    //for (int i = 0; i < 100; i++) {
    //    cout << data[i] << endl;
    //}
    std::vector<int32_t> sum(1);

    cltune::Tuner tuner(size_t{1}, size_t{0});

    // Adds the kernel. The total number of threads (the global size) is equal to 'kVectorSize', and
    // the base number of threads per work-group/thread-block (the local size) is 1. This number is
    // then multiplied by the 'GROUP_SIZE' parameter, which can take any of the specified values.
    auto id = tuner.AddKernelFromString(full_kernel, "reduction_sum_scalar_complete2", {(unsigned long)a.physical_capacity()}, {(unsigned long)group_size});

    // Sets the function's arguments
    tuner.AddArgumentInput(data);
    tuner.AddArgumentInput(sum);

    // Starts the tuner
    tuner.SetNumRuns(100);
    tuner.Tune();

    // Prints the results to screen
    //tuner.PrintToScreen();
    vector<pair<string,string>> descriptions;
    descriptions.push_back(make_pair("packingParam", "2"));

    tuner.PrintJSON("fydp-tune.json", descriptions);
    ifstream jstream{ "fydp-tune.json" };
    string test_output{ istreambuf_iterator<char>(jstream), istreambuf_iterator<char>() };
    string error;
    json11::Json json = json11::Json::parse(test_output, error, json11::JsonParse::STANDARD);

    return json["results"][0]["time"].number_value();
}


int main() {

    vector<int> v = {32, 4, 8, 16, 2};
    vector<pair<int,double>> times;
    for (size_t i = 0; i < v.size(); i++) {
        //packed_sum(v[i]);
        times.push_back(make_pair(v[i], reduce(v[i])));
    }
    sort(times.begin(), times.end(), [](const pair<int, double>& a, const pair<int, double>& b) -> bool { 
        return a.second < b.second; 
    });

    cout << "Summary:\nBitsize\tTime in ms" << endl;
    for (size_t i = 0; i < times.size(); i++) {
        cout << times[i].first << "\t" << times[i].second << endl;
    }
    return 0;
}
