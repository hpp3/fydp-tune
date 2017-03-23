__kernel void myGEMM2(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    const int row = get_local_id(0); // Local row ID (max: GROUP_SIZE_A)
    const int col = get_local_id(1); // Local col ID (max: GROUP_SIZE_A)
    const int globalRow = GROUP_SIZE_A*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = GROUP_SIZE_A*get_group_id(1) + col; // Col ID of C (0..N)

    float acc = 0.0f;
    
    const int numTiles = K/GROUP_SIZE_A;
    for (int t=0; t<numTiles; t++) {
        INIT_VAR_2D_row_major_a(A, A_sub, t, M)
        INIT_VAR_2D_col_major_b(B, B_sub, t, K)
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k<GROUP_SIZE_A; k++) {
            acc += A_sub[k][row] * B_sub[col][k];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}
