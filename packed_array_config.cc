#include "packed_array_config.h"
#include <cmath>
#include <sstream>

using namespace std;

int createMask(int width) {
    if (width == 32) {
      return -1;
    }
    return (1 << width) - 1;
}

int log2(int x) {
    int v = x;
    int r = 0;
    while ((v >>= 1) != 0) {
      r++;
    }
    return r;
}

string replaceString(string subject, const string& search,
                          const string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
    return subject;
}

PackedArrayConfig::PackedArrayConfig(string name, int requested_bit_width, 
        int section_count, int requested_indices_per_section) {
    this->name = name;

    if (requested_bit_width <= 0) {
        throw "requested_bit_width must be greater than 0";
    } else if (requested_bit_width > 32) {
        throw "requested_bit_width must be less than or equal to 32";
    } else if (section_count < 0) {
        throw "section_count must be greater than or equal 0";
    } else if (requested_indices_per_section < 0) {
        throw "requested_indices_per_section must be greater than or equal to 0";
    }

    if (32 % requested_bit_width == 0) {
        this->bitwidth = requested_bit_width;
    } else {
        this->bitwidth = 1 << (log2(requested_bit_width) + 1);
    }
   
    this->indices_per_cell = 32 / bitwidth;
    this->indices_per_cell_log2 = log2(indices_per_cell);
    this->index_mask = createMask(indices_per_cell_log2);
    this->bitwidth_log2 = log2(bitwidth);
    this->value_mask = createMask(bitwidth);

    // determine section parameters
    this->section_count = section_count;
    this->requested_indices_per_section = requested_indices_per_section;
    this->cells_per_section = (int) ceil(requested_indices_per_section / (double) indices_per_cell);

    this->cell_count = section_count * cells_per_section;
    this->index_count = cell_count * indices_per_cell; 
}

string PackedArrayConfig::generateOpenCLCode(bool prefetch, int work_group_size) const {
    stringstream ss;

    // constants
    ss << "#define name_GROUP_SIZE " << work_group_size << endl;
    ss << "#define name_bitwidth " << bitwidth << endl;
    ss << "#define name_bitwidthLog2 " << bitwidth_log2 << endl;
    ss << "#define name_valuesPerCell " << indices_per_cell << endl;
    ss << "#define name_valuesPerCellLog2 " << indices_per_cell_log2 << endl;
    ss << "#define name_valueMask 0x" << hex << value_mask;
    ss << endl;
    ss << "#define name_indexMask 0x" << hex << index_mask;
    ss << endl;
    if (prefetch) {
      // ss << "#define assignments_scope __local", << endl;
      ss << "#define name_base_offset 0" << endl;
    } else {
      // ss << "#define name_scope const __global", << endl;
      ss << "#define name_base_offset NUM_VARIABLES_ALIGNED" << endl;
    }

    // accessor functions
    ss << "uint name_cell(const uint index) { return index >> name_valuesPerCellLog2; }" << endl;
    ss << 
        "uint name_subcell(const uint index) { return (index & name_indexMask) << name_bitwidthLog2; }"
       << endl;
    ss <<
        "int name_get2(SCOPE const uint* const a, const uint cell, const uint subcell) { return (a[cell] >> subcell) & name_valueMask; }"
       << endl;

    // mutator functions
    if (bitwidth == 32) {
        ss << 
          "void name_set(SCOPE uint* const a, const uint index, const int v) { atomic_cmpxchg(&a[index], a[index], v); }"
           << endl;
    } else {
        ss << 
          "void name_set2(SCOPE uint* const a, const uint cell, const uint subcell, const int v) { "
              // the code we need to re-write in a thread-safe manner
              // "/* a[cell] &= ~(name_valueMask << subcell); a[cell] |= (v << subcell); */ "
              // lock-free looping
              << "uint original = 0; uint current = 0; " // initial values
              << "do { " // keep looping until we win
              << "    original = a[cell]; "
              << "    const uint cleared = original & ~(name_valueMask << subcell); "
              << "    const uint desired = cleared | v << subcell; "
              << "    current = atomic_cmpxchg(&a[cell], original, desired); "
              << "} while (current != original);" << "}" << endl;
        ss <<
          "void name_set(SCOPE uint* const a, const uint index, const uint v) { name_set2(a, name_cell(index), name_subcell(index), v); }"
           << endl;
    }

    // initialize local variable name to use prefetching (or not)
    if (prefetch) {
        ss <<
          "int name_get(SCOPE const uint* const a, const uint index) { const uint x = index % INDICES_PER_SECTION; return name_get2(a, name_cell(x), name_subcell(x)); }"
           << endl;
        ss << "#define LINEAR_LOCAL_ID_name  (get_local_id(1) * get_local_size(0) + get_local_id(0))"
           << endl;
      ss << "#define LINEAR_GROUP_ID_name  (get_group_id(1) * get_num_groups(0) + get_group_id(0))"
          << endl;
      ss << 
          "#define LINEAR_GLOBAL_ID_name  (LINEAR_GROUP_ID_name * WORKGROUPSIZE + LINEAR_LOCAL_ID_name)"
          << endl;
      ss << 
          "uint name_getSectionID() { return (get_group_id(0) * NUMBER_OF_SECTIONS) / get_num_groups(0); }"
          << endl;
      ss << "uint name_getSectionID_2D() { return LINEAR_GROUP_ID_name % NUMBER_OF_SECTIONS;}" << endl;
      ss << "#define INIT_VAR_name(global,local) __local uint (local)[CELLS_PER_SECTION]; "
          << "prefetch_name(name_getSectionID(), get_local_id(0), (global), (local));" << endl;
      ss << "#define INIT_VAR_2D_name(global,local) __local uint (local)[CELLS_PER_SECTION]; "
         <<  "prefetch_name(name_getSectionID_2D(), LINEAR_LOCAL_ID_name, (global), (local));" << endl;
      ss << 
          "#define INIT_VAR_2D_row_major_name(global,local, tileID, M) __local uint (local)[WORKGROUPSIZE][WORKGROUPSIZE]; "
              << "prefetch_2D_row_major_name(tileID, M, (global), (local));" << endl;
      ss << 
          "#define INIT_VAR_2D_col_major_name(global,local, tileID, K) __local uint (local)[WORKGROUPSIZE][WORKGROUPSIZE]; "
              << "prefetch_2D_col_major_name(tileID, K, (global), (local));" << endl;
    } else {
      ss << 
          "int name_get(SCOPE const uint* const a, const uint index) { return name_get2(a, name_cell(index), name_subcell(index)); }"
          << endl;
      ss << "#define INIT_VAR_name(global,local) __global const uint* (local) = (global);" << endl;
      ss << "#define INIT_VAR_2D_name(global,local) __global const uint* (local) = (global);" << endl;
      ss << 
          "#define INIT_VAR_2D_row_major_name(global,local, tileID, M) __global const uint (*(local))[WORKGROUPSIZE] = (global);"
          << endl;
      ss << 
          "#define INIT_VAR_2D_col_major_name(global,local, tileID, K) __global const uint (*(local))[WORKGROUPSIZE] = (global);"
          << endl;
    }

    // prefetch functions
    int loop_bound_floor = (int) floor((float) cells_per_section / work_group_size);
    int loop_bound_ceil = (int) ceil((float) cells_per_section / work_group_size);
    if (prefetch) {

      // translate section index to global index
      ss << "uint name_global_index(uint sectionID, uint local_index) {" << endl;
      ss << "  return (sectionID * INDICES_PER_SECTION + local_index);" << endl;
      ss << "}" << endl;

      ss << 
          "void prefetch_name(const uint sectionID, const uint threadID, __global const uint* const ga, __local uint* la) {"
          << endl;

      // calculate g_offset as the index from which to start prefetching
      ss << "  const uint g_offset = sectionID * CELLS_PER_SECTION; " << endl;

      // split the loop in two: first where we know every iteration has data; second where we need
      // to check bounds for these iterations we know that we are in bounds
      if (loop_bound_floor > 0) {
        ss << "  #pragma unroll LOOP_BOUND_FLOOR" << endl;
      }
      ss << 
          "  for (uint loop=0, cell=threadID; loop < LOOP_BOUND_FLOOR; loop++, cell += WORKGROUPSIZE) {"
          << endl;
      ss << "    const uint g_cell = g_offset + cell; " << endl;
      ss << "    la[cell] = ga[g_cell]; " << endl;
      ss << "  }" << endl;
      if (loop_bound_floor < loop_bound_ceil) {
        // for these iterations we need to check that we are in bounds
        // there will be at most one iteration of this loop
        ss << "  #pragma unroll 1" << endl;
        ss << 
            "  for (uint loop=LOOP_BOUND_FLOOR, cell= WORKGROUPSIZE * LOOP_BOUND_FLOOR + threadID; loop < LOOP_BOUND_CEIL; loop++, cell += WORKGROUPSIZE) {"
            << endl;
        ss << "    const uint g_cell = g_offset + cell;" << endl;
        ss << "    if (cell < CELLS_PER_SECTION) { la[cell] = ga[g_cell]; }" << endl;
        ss << "  }" << endl;
      }
      ss << "  barrier(CLK_LOCAL_MEM_FENCE);" << endl;
      ss << "}" << endl;

      ss << 
          "void prefetch_2D_row_major_name(const uint tileID, const uint M, __global const uint* const ga, __local uint* la) {"
          << endl;

      ss << "const int row = get_local_id(0);" << endl;
      ss << "const int col = get_local_id(1);" << endl;
      ss << "const int globalRow = WORKGROUPSIZE * get_group_id(0) + row;" << endl;
      ss << "const int tiledCol = WORKGROUPSIZE * tileID + col;" << endl;
      ss << "la[col*WORKGROUPSIZE + row] = ga[tiledCol*M + globalRow];" << endl;
      ss << "}" << endl;

      ss << 
          "void prefetch_2D_col_major_name(const uint tileID, const uint K, __global const uint* const ga, __local uint* la) {"
          << endl;

      ss << "const int row = get_local_id(0);" << endl;
      ss << "const int col = get_local_id(1);" << endl;
      ss << "const int globalCol = WORKGROUPSIZE * get_group_id(1) + col;" << endl;
      ss << "const int tiledRow = WORKGROUPSIZE * tileID + row;" << endl;
      ss << "la[col*WORKGROUPSIZE + row] = ga[globalCol*K + tiledRow];" << endl;
      ss << "}" << endl;

    }

    int actual_indices_per_section = cells_per_section * indices_per_cell;
    string output = ss.str();
    output = replaceString(output, "name", name);
    output = replaceString(output, "INDICES_PER_SECTION", to_string(actual_indices_per_section));
    output = replaceString(output, "NUMBER_OF_SECTIONS", to_string(section_count));
    output = replaceString(output, "CELLS_PER_SECTION", to_string(cells_per_section));
    output = replaceString(output, "WORKGROUPSIZE", to_string(work_group_size));
    output = replaceString(output, "LOOP_BOUND_FLOOR", to_string(loop_bound_floor));
    output = replaceString(output, "LOOP_BOUND_CEIL", to_string(loop_bound_ceil));
    output = replaceString(output, "SCOPE", prefetch ? "__local" : "__global");
    return output;
}
