#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include "json11.hpp"
#include "packed_array_impl.h"

#define TUNE_START(bitsizes) \
    std::vector<pair<int,double>> cpack_times;                                                                                  \
    for (auto cpack_bitsize: bitsizes) {                                                                                        \
        std::string cpack_header = "";
#define TUNE_END(tuner) \
        tuner.PrintJSON("cpack_tmp.json", {});                                                                                  \
        std::ifstream cpack_json_file{ "cpack_tmp.json" };                                                                      \
        std::string cpack_json_string{ std::istreambuf_iterator<char>(cpack_json_file), std::istreambuf_iterator<char>() };     \
        std::string cpack_error;                                                                                                \
        json11::Json cpack_json = json11::Json::parse(cpack_json_string, cpack_error, json11::JsonParse::STANDARD);             \
        cpack_times.push_back(std::make_pair(cpack_bitsize, cpack_json["results"][0]["time"].number_value()));                  \
    }                                                                                                                           \
    std::sort(cpack_times.begin(), cpack_times.end(), [](const pair<int, double>& a, const pair<int, double>& b) -> bool {      \
        return a.second < b.second;                                                                                             \
    });                                                                                                                         \
    std::cout << "Summary:\nBitsize\tTime in ms" << std::endl;                                                                  \
    for (size_t i = 0; i < cpack_times.size(); i++) {                                                                           \
        std::cout << cpack_times[i].first << "\t" << cpack_times[i].second << std::endl;                                        \
    }
#define PROCESS_VEC(vec, name, do_pack, group_size) \
    {                                                                                                                           \
        PackedArrayImpl cpack_packer(name, cpack_bitsize, vec.size());                                                          \
        int32_t *cpack_cells = cpack_packer.getCells();                                                                         \
        std::vector<int32_t> cpack_tmp(cpack_cells, cpack_cells+cpack_packer.physical_capacity());                              \
        for (size_t i = 0; i < vec.size(); i++) {                                                                               \
            cpack_packer.set(i, vec[i]);                                                                                        \
        }                                                                                                                       \
        vec = std::move(cpack_tmp);                                                                                             \
        cout << vec.size() << endl;\
        cpack_header += cpack_packer.getConfig().generateOpenCLCode(do_pack, group_size);                                       \
    }
#define KERNEL_STRING(kernel_file) \
    cpack_kernel_string(kernel_file, cpack_header)

string cpack_kernel_string(string kernel_file, string cpack_header) {
    std::ifstream cpack_kernel_file{ kernel_file };
    std::string cpack_kernel{ std::istreambuf_iterator<char>(cpack_kernel_file), std::istreambuf_iterator<char>() };
    return cpack_header + cpack_kernel;
}
