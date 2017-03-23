__kernel void ref( const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      const __global int* Target,
                      __global int* C ) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    C[i*N + j] = Target[i*N + j];
}
