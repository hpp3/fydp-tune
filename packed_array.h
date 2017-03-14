#ifndef _PACKED_ARRAY_H_
#define _PACKED_ARRAY_H_

#include <string>
using namespace std;

class PackedArray {

public:
    /** The number of logical elements in this PackedArray. */
    virtual int32_t size() const = 0;

    /** The logical capacity of this PackedArray. */
    virtual int32_t capacity() const = 0;

    /**
    * The maximum value that can be stored in this packed array. If the bitwidth is 32 this will say
    * int32_teger.MAX_VALUE (which is only 31 bits).
    */
    virtual int32_t maxValue() const = 0;

    /* accessors of array elements */
    virtual void set(const int32_t index, const int32_t value) = 0;
    virtual int32_t get(int32_t index) const = 0;


    virtual string toString() {
        string b;
        b += "[";

        if (size() > 0)
            b += get(0);

        for (int32_t i = 1; i < size(); i++) {
            b += ',';
            b += get(i);
        }

        b += "]";
        return b;
    }
};

#endif
