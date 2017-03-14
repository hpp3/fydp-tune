/** Performs a sum reduction entire in global memory. */
__kernel void reduction_sum_scalar1(
        __global int* input,
        __global int* output
    )
{
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // Pre-fetch data to local memory
    output[tid] = input[tid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = group_size/2; i > 0; i >>= 1) {
        if (lid < i)
            output[tid] += output[tid + i];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Copy result from local to global memory
    if (lid == 0)
        output[get_group_id(0)] = output[tid];
}

/** Performs a sum reduction using local memory. */
__kernel void reduction_sum_scalar2(
                                    __local int* partial_sums,
                                    __global int* input,
                                    __global int* output
                                    )
{
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);

    // Pre-fetch data to local memory
    partial_sums[lid] = input[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = group_size/2; i > 0; i >>= 1) {
        if (lid < i)
            partial_sums[lid] += partial_sums[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Copy result from local to global memory
    if (lid == 0)
        output[get_group_id(0)] = partial_sums[0];
}

/** Performs a sum reduction using local memory. */
__kernel void reduction_sum_scalar_complete2(
                                    __global int* input,
                                    __global int* sum
                                    )
{
    int lid = get_local_id(0);
    __local int partial_sums[128];
    int group_size = get_local_size(0);
    partial_sums[lid] = input[get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = group_size/2; i > 0; i >>= 1) {
        if (lid < i)
            partial_sums[lid] += partial_sums[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        *sum = partial_sums[0];
        //printf("[%d] sum=%d\n", get_global_id(0), *sum);
    }
}
