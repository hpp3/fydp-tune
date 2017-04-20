__kernel void multi(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      const __global int* Target,
                      __global int* C) {
    
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    int acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += _2D_arrA_get(A, globalRow, k) * _2D_arrB_get(B, k, globalCol);
    }
 
    C[globalCol + globalRow*N] = acc;
}
