#ifndef _PACKED_2D_ARRAY_IMPL_H_
#define _PACKED_2D_ARRAY_IMPL_H_

#include <atomic>
#include <string>
#include <sstream>
#include <cmath>
#include "packed_array.h"
#include "packed_array_config.h"
#include "packed_array_section.h"
#include "packed_array_impl.h"

using namespace std;

class Packed2DArrayImpl {
public:
	Packed2DArrayImpl(const string name, const int32_t requested_bitwidth, const int32_t height, const int32_t width, bool row_major) 
		: Packed2DArrayImpl(name, requested_bitwidth, 1, height, width, row_major) {
	}

	Packed2DArrayImpl(const string name, const int32_t requested_bitwidth, const int32_t section_count, const int32_t height, const int32_t width, bool row_major) 
		: impl(PackedArrayImpl(name, requested_bitwidth, section_count, height*width, row_major)), name(name), width(width), height(height), row_major(row_major) {
	}


	string generateOpenCLCode(bool prefetch, int group_size) {
		stringstream ss;
		ss << impl.getConfig().generateOpenCLCode(prefetch, group_size);
		ss << "#define " << name << "_width " << width << endl;
		ss << "#define " << name << "_height " << height << endl;
		if (row_major) {
			ss << "#define _2D_" << name << "_get(arr, i, j) " << name << "_get(arr, i*" << name << "_width+j)" << endl;
		} else {
			ss << "#define _2D_" << name << "_get(arr, i, j) " << name << "_get(arr, j*" << name << "_height+i)" << endl;
		}
		return ss.str();
	}

    void recalibrateSize() {
		impl.recalibrateSize();
    }

    int32_t size() const {
        return impl.size();
    }

    int32_t capacity() const {
        return impl.capacity();
    }

    int32_t physical_size() const {
		return impl.physical_size();
    }

    int32_t physical_capacity() const {
		return impl.physical_capacity();
    }

    int32_t maxValue() const {
		return impl.maxValue();
    }

    int32_t cellCount() {
		return impl.cellCount();
    }

    int32_t cellCount(const int32_t capacity) {
		return impl.cellCount(capacity);
    }

    void append(const int32_t value) {
		impl.append(value);
    }

    void set(const int32_t y, const int32_t x, const int32_t value) {
		if (row_major) {
			impl.set(y*width + x, value);
		} else {
			impl.set(x*height + y, value);
		}
    }

    int32_t get(const int32_t y, const int32_t x) const {
		if (row_major) {
			return impl.get(y*width + x);
		} else {
			return impl.get(x*height+ y);
		}
    }


    int32_t* getCells() {
		return impl.getCells();
    }

private:
	PackedArrayImpl impl;
	string name;
	int32_t width;
	int32_t height;
	bool row_major;
};

#endif
