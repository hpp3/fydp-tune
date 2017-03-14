#ifndef _PACKED_ARRAY_CONFIG_H_
#define _PACKED_ARRAY_CONFIG_H_

#include <iostream>
#include <string>

using namespace std;

class PackedArrayConfig {
    public:
        string name;
        int bitwidth;
        int bitwidth_log2;

        int indices_per_cell;
        int indices_per_cell_log2;

        int index_mask;
        int value_mask;

        int index_count;
        int section_count;

        int requested_indices_per_section;
        int cells_per_section;
        int cell_count;


    public:
        PackedArrayConfig(string name, int requested_bit_width, 
                int section_count=1, int requested_indices_per_section=1);

        string generateOpenCLCode(bool prefetch, int workgroupsize) const;
};

#endif
