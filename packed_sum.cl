__kernel void packed_sum(
    __global const uint* const packed_input, 
    __global uint* const sum)
{
    const int tid = get_global_id(0);

    INIT_VAR_test(packed_input, input)
    const int bit = test_get(input, tid);

    atomic_add(&sum[0], bit);
}

