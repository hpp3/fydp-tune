__kernel void conv_prefetch(
    const __global int *img, 
    __constant int *filter,
    int filter_size,
    const __global int *target,
    __global int *output
) {

    INIT_VAR_2D_imgArr(img, localArr)

    const int globalCol = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int globalNumRow = get_global_size(0);

    const int half_filter_size = filter_size/2;
    // ^ same as filter_size - 1, fitler size is assumed to be odd

    const int localCol = get_local_id(0) + half_filter_size;
    const int localRow = get_local_id(1) + half_filter_size;
    const int localNumRow = 2*half_filter_size + get_local_size(1);

    int sum = 0;
    int fIdx = 0;
    for (int c = -half_filter_size; c <= half_filter_size; c++) {
        for (int r = -half_filter_size; r <= half_filter_size; r++, fIdx++) {
            int finalCol = localCol + c;
            int finalRow = localRow + r;
            sum += localArr[finalCol][finalRow] * filter[ fIdx ]; 
        }
    }
    output[globalCol*globalNumRow+globalRow] = sum;
}
