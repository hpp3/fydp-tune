__kernel void conv_simple(
    const __global int *img, 
    __constant int *filter,
    int filter_size,
    const __global int *target,
    __global int *output
) {

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int globalNumCol = get_global_size(0);
    const int globalNumRow = get_global_size(1);
    const int half_filter_size = filter_size/2;

    int sum = 0;
    int fIdx = 0;
    for (int c = -half_filter_size; c <= half_filter_size; c++) {
        for (int r = -half_filter_size; r <= half_filter_size; r++, fIdx++) {
            int finalCol = col + c;
            int finalRow = row + r;

            if ( 0 <= finalCol && finalCol < globalNumCol && 0 <= finalRow && finalRow < globalNumRow )
                sum += img[finalCol*globalNumRow+finalRow] * filter[fIdx];
        }
    }

    output[col*globalNumRow+row] = sum;
}
