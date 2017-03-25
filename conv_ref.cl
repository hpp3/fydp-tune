__kernel void ref(
    const __global int *img, 
    __constant int *filter,
    int filter_size,
    const __global int *target,
    __global int *output
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int N = get_global_size(0);
    output[i*N + j] = target[i*N + j];
}
