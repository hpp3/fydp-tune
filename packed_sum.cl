__kernel void packed_sum(
    __global const uint* const packed_input, 
    __global uint* const sum)
{
    const int tid = get_global_id(0);
    const int bit = test_get(packed_input, tid);

    atomic_add(&sum[0], bit);
}

