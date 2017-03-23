__kernel void multi_tile(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      const __global int* Target,
                      __global int* C) {
    
    const int row = get_local_id(0); // Local row ID (max: arrA_GROUP_SIZE)
    const int col = get_local_id(1); // Local col ID (max: arrA_GROUP_SIZE)
    const int globalRow = arrA_GROUP_SIZE*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = arrA_GROUP_SIZE*get_group_id(1) + col; // Col ID of C (0..N)

    int acc = 0.0f;
    
    const int numTiles = K/arrA_GROUP_SIZE;
    for (int t=0; t<numTiles; t++) {
        INIT_VAR_2D_row_major_arrA(A, A_sub, t, M)
        INIT_VAR_2D_col_major_arrB(B, B_sub, t, K)
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k<arrA_GROUP_SIZE; k++) {
            acc += A_sub[k][row] * B_sub[col][k];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}
