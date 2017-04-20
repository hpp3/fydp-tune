#ifndef _PACKED_ARRAY_SECTION_H_
#define _PACKED_ARRAY_SECTION_H_

#include "packed_array.h"
#include <cassert>
using namespace std;

class PackedArraySection : public PackedArray {

public:
    PackedArraySection(PackedArray &delegate, const int32_t start_index,
        const int32_t requested_indices_per_section, const int32_t actual_indices_per_section) :

        delegate( delegate ), start_index( start_index ), actual_indices_per_section( actual_indices_per_section ), 
        requested_indices_per_section( requested_indices_per_section ) {

        assert(start_index + actual_indices_per_section <= delegate.capacity() 
            && "backing PackedArray does not have capacity to support this view" );
    }

    int32_t size() const {
        return requested_indices_per_section;
    }

    int32_t capacity() const {
        return actual_indices_per_section;
    }

    int32_t maxValue() const {
        return delegate.maxValue();
    }

    void set(const int32_t index, const int32_t value) {
        // index int32_to the backing PackedArray
        assert(index < actual_indices_per_section);
        delegate.set(start_index + index, value);
    }

    int32_t get(const int32_t index) const {
        // index int32_to the backing PackedArray
        assert( index < actual_indices_per_section && "index exceeds sectionCapacity: " );
        return delegate.get(start_index + index);
    }

private:

    PackedArray &delegate;
    const int32_t start_index;

    const int32_t actual_indices_per_section;
    const int32_t requested_indices_per_section;

};

#endif
