__kernel void conv_raw_prefetch(
    const __global int *img, 
    __constant int *filter,
    int filter_size,
    const __global int *target,
    __global int *output
) {

    const int globalCol = get_global_id(0);
    const int globalRow = get_global_id(1);
    const int half_filter_size = filter_size/2;

    const int col = get_local_id(0);
    const int row = get_local_id(1);
    const int cacheCol = col + half_filter_size;
    const int cacheRow = row + half_filter_size;

    __global const int (*ga)[GLOBAL_NUM_ROW] = img;
    __local int la[LOCAL_NUM_COL+2*HALF_FILTER_SIZE][LOCAL_NUM_ROW+2*HALF_FILTER_SIZE];

    la[cacheCol][cacheRow] = ga[globalCol][globalRow];

    if ( row < half_filter_size ) {
        int edgeRow = cacheRow - half_filter_size;
        int edgeGlobalRow = globalRow - half_filter_size;
        if ( 0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW )
           la[cacheCol][edgeRow] = ga[globalCol][edgeGlobalRow];
        else
           la[cacheCol][edgeRow] = 0;
    }

    if ( row >= LOCAL_NUM_ROW - half_filter_size ) {
        int edgeRow = cacheRow + half_filter_size;
        int edgeGlobalRow = globalRow + half_filter_size;
        if ( 0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW )
           la[cacheCol][edgeRow] = ga[globalCol][edgeGlobalRow];
        else
           la[cacheCol][edgeRow] = 0;
    }

    if ( col < half_filter_size ) {
        int edgeCol = cacheCol - half_filter_size;
        int edgeGlobalCol = globalCol - half_filter_size;
        if ( 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][cacheRow] = ga[edgeGlobalCol][globalRow];
        else
           la[edgeCol][cacheRow] = 0;
    }

    if ( col >= LOCAL_NUM_COL - half_filter_size ) {
        int edgeCol = cacheCol + half_filter_size;
        int edgeGlobalCol = globalCol + half_filter_size;
        if ( 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][cacheRow] = ga[edgeGlobalCol][globalRow];
        else
           la[edgeCol][cacheRow] = 0;
    }

    if ( row < half_filter_size && col < half_filter_size ) {
        int edgeRow = cacheRow - half_filter_size;
        int edgeGlobalRow = globalRow - half_filter_size;
        int edgeCol = cacheCol - half_filter_size;
        int edgeGlobalCol = globalCol - half_filter_size;
        if (  0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW && 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][edgeRow] = ga[edgeGlobalCol][edgeGlobalRow];
        else
           la[edgeCol][edgeRow] = 0;
    }
    if ( row >= LOCAL_NUM_ROW - half_filter_size && col >= LOCAL_NUM_COL - half_filter_size) {
        int edgeRow = cacheRow + half_filter_size;
        int edgeGlobalRow = globalRow + half_filter_size;
        int edgeCol = cacheCol + half_filter_size;
        int edgeGlobalCol = globalCol + half_filter_size;
        if (  0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW && 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][edgeRow] = ga[edgeGlobalCol][edgeGlobalRow];
        else
           la[edgeCol][edgeRow] = 0;
    }
    if ( row < half_filter_size  && col >= LOCAL_NUM_COL - half_filter_size) {
        int edgeRow = cacheRow - half_filter_size;
        int edgeGlobalRow = globalRow - half_filter_size;
        int edgeCol = cacheCol + half_filter_size;
        int edgeGlobalCol = globalCol + half_filter_size;
        if (  0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW && 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][edgeRow] = ga[edgeGlobalCol][edgeGlobalRow];
        else
           la[edgeCol][edgeRow] = 0;
    }
    if ( row >= LOCAL_NUM_ROW - half_filter_size && col < half_filter_size) {
        int edgeRow = cacheRow + half_filter_size;
        int edgeGlobalRow = globalRow + half_filter_size;
        int edgeCol = cacheCol - half_filter_size;
        int edgeGlobalCol = globalCol - half_filter_size;
        if (  0 <= edgeGlobalRow && edgeGlobalRow <  GLOBAL_NUM_ROW && 0 <= edgeGlobalCol && edgeGlobalCol <  GLOBAL_NUM_COL )
           la[edgeCol][edgeRow] = ga[edgeGlobalCol][edgeGlobalRow];
        else
           la[edgeCol][edgeRow] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int sum = 0;
    int fIdx = 0;
    for (int c = -half_filter_size; c <= half_filter_size; c++) {
        for (int r = -half_filter_size; r <= half_filter_size; r++, fIdx++) {
            int finalCol = cacheCol + c;
            int finalRow = cacheRow + r;
            sum += la[finalCol][finalRow] * filter[ fIdx ]; 
        }
    }
    output[globalCol*GLOBAL_NUM_ROW+globalRow] = sum;
}
