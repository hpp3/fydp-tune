#include <vector>
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>

#include "cltune.h"
#include "packed_array_impl.h"
#include "json11.hpp"
#include "cpack.h"

using namespace std;

int main() {
    int bitsize = 32;
    int simpleSize = 10*10;
    PackedArrayImpl a("a", bitsize, simpleSize);
    PackedArrayImpl b("b", bitsize, simpleSize);

    int group_size = 32;

    // read the file
    ifstream infile{ "multi_tile.cl" };
    string kernel{ istreambuf_iterator<char>(infile), istreambuf_iterator<char>() };
    string header_a = a.getConfig().generateOpenCLCode(true, group_size);
    string header_b = b.getConfig().generateOpenCLCode(true, group_size);
    cout << header_a + header_b + kernel << endl;
    return 0;
}
