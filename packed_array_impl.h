#ifndef _PACKED_ARRAY_IMPL_H_
#define _PACKED_ARRAY_IMPL_H_

#include <atomic>
#include <string>
#include <cmath>
#include "packed_array.h"
#include "packed_array_config.h"
#include "packed_array_section.h"
using namespace std;
/**
 * Stores multiple logical values in a single cell of an int32_t array. "Index" refers to the logical
 * index of the value. "Cell" refers to the physical index, or to the bits in a cell.
 *
 */
class PackedArrayImpl : public PackedArray {
public:

    /**
    * Construct a PackedArray with a generated name. The actual bitwidth will be a power of two that
    * is greater than or equal to the requested bitwidth. The actual logical capacity will be at
    * least as large as the requested capacity.
    *
    * @param requested_bitwidth
    * @param requested_capacity
    */
    PackedArrayImpl(const int32_t requested_bitwidth, const int32_t requested_capacity)
        : PackedArrayImpl("PackedArray" + to_string((uid=-1)++), requested_bitwidth, requested_capacity) {
    }

    PackedArrayImpl(const string name, const int32_t requested_bitwidth, const int32_t requested_capacity)
        : PackedArrayImpl(name, requested_bitwidth, 1, requested_capacity) {
    }

    PackedArrayImpl(const string name, const int32_t requested_bitwidth, const int32_t section_count, const int32_t indices_per_section, bool row_major=true)
        : PackedArrayImpl(PackedArrayConfig(name, requested_bitwidth, section_count, indices_per_section, row_major)) {
    }

    PackedArrayImpl(const PackedArrayConfig &config)
            : highest_set_index(-1), uid(1), config( config ) {
        // allocate the cells, set them to zero
        cells = new int32_t[config.cell_count]();
        cell_len = config.cell_count;
    }

    ~PackedArrayImpl() {
        delete []cells;
    }

    const PackedArrayConfig& getConfig() {
        return config; 
    }

    /** Recompute the highest_set_index if the underlying array is directly edited **/
    void recalibrateSize() {
        int32_t i = 0;
        for (i = cell_len - 1; i >= 0; i--) {
            if (cells[i] != 0)
                break;
        }
        if (i == -1) {
            highest_set_index = -1;
            return;
        }
        int32_t index = (i + 1) * config.indices_per_cell - 1;
        while (get(index) == 0) {
            index--;
        }
        highest_set_index = index;
    }

    /** The number of logical elements in this PackedArray. */
    int32_t size() const {
        return highest_set_index + 1;
    }

    /** The logical capacity of this PackedArray. */
    int32_t capacity() const {
        return config.index_count;
    }

    /** Physical Usable Capacity **/
    int32_t physical_size() const {
        return ceil( double(size()) / config.indices_per_cell );
    }

    /** Physical Usable Capacity **/
    int32_t physical_capacity() const {
        return ceil( double(capacity()) / config.indices_per_cell );
    }


    /**
    * The maximum value that can be stored in this packed array. If the bitwidth is 32 this will say
    * int32_teger.MAX_VALUE (which is only 31 bits).
    */
    int32_t maxValue() const {
        return (config.bitwidth < 32) ? config.value_mask : (2*31-1);
    }

    int32_t cellCount() {
        const int32_t x = cellCount(config.index_count);
        assert(x == cell_len);
        return x;
    }

    int32_t cellCount(const int32_t capacity) {
        return cellCount(config.index_count, config.indices_per_cell);
    }

    static int32_t cellCount(const int32_t capacity, const int32_t values_per_cell) {
        return (int32_t) ceil((float) capacity / (float) values_per_cell);
    }

    void append(const int32_t value) {
        set(highest_set_index + 1, value);
    }

    void set(const int32_t index, const int32_t value) {
        // the mask for 32 bits (0xFFFFFFFF) is -1, so in that case we can't check value <= mask
        assert(value <= config.value_mask || config.bitwidth == 32);
        const int32_t cell = this->cell(index);
        assert(cell < cell_len);
        const int32_t subcell = this->subcell(index);
        set(cell, subcell, value);
        if (index > highest_set_index) {
              highest_set_index = index;
        }
    }

    int32_t get(const int32_t index) const {
        return get(this->cell(index), this->subcell(index));
    }


    /**
    * Returns a reference to the underlying data. Clients should not mutate the data. Reference is
    * made available to facilitate serialization.
    */
    int32_t* getCells() {
        return cells;
    }

    bool hasLeftovers() {
        return (highest_set_index >= 0) && (((highest_set_index + 1) % config.indices_per_cell) != 0);
    }

    PackedArrayImpl getLeftovers() {
        const int32_t leftover_size = ((highest_set_index + 1) % config.indices_per_cell);
        PackedArrayImpl leftovers(config.name + "_leftovers", config.bitwidth, leftover_size);
        for (int32_t j = 0, i = highest_set_index - leftover_size + 1; i <= highest_set_index; i++, j++) {
            leftovers.set(j, get(i));
        }
        return leftovers;
    }

    PackedArraySection section(const int32_t section) {
        assert(section < config.section_count);
        const int32_t actualIndices_per_section = config.cells_per_section * config.indices_per_cell;
        const int32_t startIndex = section * actualIndices_per_section;
        return PackedArraySection(*this, startIndex, config.requested_indices_per_section, actualIndices_per_section);
    }


protected:
    int32_t cell(const int32_t index) const {
        // these expressions are equivalent when bitwidth is a power of 2
        // return index / values_per_cell;
        return index >> config.indices_per_cell_log2;
    }

    int32_t subcell(const int32_t index) const {
        return (index & config.index_mask) << config.bitwidth_log2;
    }

private:
    void set(const int32_t cell, const int32_t subcell, const int32_t value) {
        assert(value <= config.value_mask || config.bitwidth == 32);
        cells[cell] &= ~(config.value_mask << subcell);
        cells[cell] |= value << subcell;
    }

    int32_t get(const int32_t cell, const int32_t subcell) const {
        return ((uint32_t)cells[cell] >> (uint32_t)subcell) & config.value_mask;
    }


private:
    int32_t* cells;
    int32_t cell_len;
    int32_t highest_set_index;
    int32_t uid;

public:
    const PackedArrayConfig config;

};

#endif
