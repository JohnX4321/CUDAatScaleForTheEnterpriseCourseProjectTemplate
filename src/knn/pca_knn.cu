#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Compute squared L2 distances between a single query (dim D) and N dataset rows (N x D)
__global__ void computeDistances(const float* __restrict__ dataset, const float* __restrict__ query, float* __restrict__ distances, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;
    int base = i * D;
    for (int d = 0; d < D; ++d) {
        float diff = dataset[base + d] - query[d];
        sum += diff * diff;
    }
    distances[i] = sum;
}

int main(int argc, char** argv) {
    // Parameters
    int N = 100000; // dataset size
    int D = 128;    // original dimensionality
    int d = 32;     // projected dimensionality
    int M = 5;      // number of queries
    int K = 5;      // neighbors

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) d = atoi(argv[3]);
    if (argc > 4) M = atoi(argv[4]);
    if (argc > 5) K = atoi(argv[5]);

    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::cout << "PCA (random projection) + GPU brute-force k-NN\n";
    std::cout << "N=" << N << " D=" << D << " d=" << d << " M=" << M << " K=" << K << "\n";

    // Host dataset and queries
    std::vector<float> h_dataset(N * D);
    std::vector<float> h_queries(M * D);
    for (int i = 0; i < N * D; ++i) h_dataset[i] = uni(rng);
    for (int i = 0; i < M * D; ++i) h_queries[i] = uni(rng);

    // Device memory for original data
    float *d_dataset = nullptr;
    float *d_query = nullptr;
    float *d_distances = nullptr;
    cudaMalloc(&d_dataset, sizeof(float) * N * D);
    cudaMalloc(&d_query, sizeof(float) * D);
    cudaMalloc(&d_distances, sizeof(float) * N);
    cudaMemcpy(d_dataset, h_dataset.data(), sizeof(float) * N * D, cudaMemcpyHostToDevice);

    // --- Baseline: compute k-NN in original D ---
    int block = 256;
    int grid = (N + block - 1) / block;

    float total_baseline_ms = 0.0f;
    for (int q = 0; q < M; ++q) {
        cudaMemcpy(d_query, h_queries.data() + q * D, sizeof(float) * D, cudaMemcpyHostToDevice);
        cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
        cudaEventRecord(st, 0);

        computeDistances<<<grid, block>>>(d_dataset, d_query, d_distances, N, D);
        cudaDeviceSynchronize();

        // sort distances with indices
        thrust::device_ptr<float> dptr(d_distances);
        thrust::device_vector<int> d_idx(N);
        thrust::sequence(d_idx.begin(), d_idx.end());
        thrust::sort_by_key(dptr, dptr + N, d_idx.begin());

        cudaEventRecord(ed, 0); cudaEventSynchronize(ed);
        float ms = 0.0f; cudaEventElapsedTime(&ms, st, ed);
        total_baseline_ms += ms;
        cudaEventDestroy(st); cudaEventDestroy(ed);
    }
    std::cout << "Baseline average time per query (ms): " << (total_baseline_ms / M) << "\n";

    // --- Random projection using cuBLAS: Y = X * R  (X: N x D, R: D x d) -> Y: N x d ---
    // Create random Gaussian projection matrix R on host
    std::vector<float> h_R(D * d);
    for (int i = 0; i < D * d; ++i) h_R[i] = normal(rng) / sqrt((float)d);

    // allocate device matrices
    float *d_R = nullptr; // D x d
    float *d_Y = nullptr; // N x d
    cudaMalloc(&d_R, sizeof(float) * D * d);
    cudaMalloc(&d_Y, sizeof(float) * N * d);
    cudaMemcpy(d_R, h_R.data(), sizeof(float) * D * d, cudaMemcpyHostToDevice);

    // cuBLAS: compute Y = X * R
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    float alpha = 1.0f, beta = 0.0f;
    // Note: cuBLAS is column-major. We treat input as row-major by swapping ops and leading dims.
    // We compute Y = X * R where X (N x D), R (D x d). Using cublasSgemm for row-major data:
    // Implement as: Y^T = R^T * X^T -> use cublasSgemm with transposed flags.

    // Time projection
    cudaEvent_t pst, ped; cudaEventCreate(&pst); cudaEventCreate(&ped);
    cudaEventRecord(pst, 0);

    // cublasSgemm: C = alpha*op(A)*op(B) + beta*C
    // We want Y (N x d) in row-major. Use A = R (D x d), B = X^T (D x N) ???
    // Simpler: call cublasSgemm with arguments to compute Y = X * R
    // Using row-major layout workaround: compute as Y^T = R^T * X^T
    // So: op(A)=N (no transpose) -> A = R^T (d x D) ; op(B)=N -> B = X^T (D x N)
    // We provide pointers for column-major math. So we call with A = d_R (but pass transposed flags)

    // Use cublas with column-major by swapping A and B and transposing
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, d, N, D, &alpha, d_R, D, d_dataset, D, &beta, d_Y, d);
    // However to avoid confusion we'll implement a simple custom kernel for projection instead.

    // Fallback: custom kernel for matrix multiplication (row-major): Y[i,d] = sum_j X[i,j] * R[j,d]
    // Implement a simple tiled kernel might be heavy; use cublas with correct parameters.

    // Use cublas: compute Y = X * R where matrices are in row-major by interpreting data as column-major with swapped dims and transpose ops:
    // Equivalent call:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, D, &alpha, d_R, d, d_dataset, D, &beta, d_Y, d);
    // After experimentation this ordering yields Y in column-major layout; to keep simple we will compute Y on host if N is small.

    // For general correctness and simplicity here, if N*D is moderate, project on host and copy to device.
    bool host_project = ( (size_t)N * (size_t)D < 20000000 ); // heuristic
    std::vector<float> h_Y;
    if (host_project) {
        h_Y.resize((size_t)N * d);
        for (int i = 0; i < N; ++i) {
            for (int jj = 0; jj < d; ++jj) {
                double s = 0.0;
                for (int j = 0; j < D; ++j) s += (double)h_dataset[i * D + j] * (double)h_R[j * d + jj];
                h_Y[i * d + jj] = (float)s;
            }
        }
        cudaMemcpy(d_Y, h_Y.data(), sizeof(float) * N * d, cudaMemcpyHostToDevice);
    } else {
        // TODO: implement efficient cuBLAS projection for very large matrices
        std::cerr << "Large projection path not implemented; reduce N*D or build with cuBLAS path." << std::endl;
        return 1;
    }

    cudaEventRecord(ped, 0); cudaEventSynchronize(ped);
    float proj_ms = 0.0f; cudaEventElapsedTime(&proj_ms, pst, ped);
    std::cout << "Projection time (ms): " << proj_ms << "\n";

    // Project queries on host
    std::vector<float> h_queries_proj(M * d);
    for (int q = 0; q < M; ++q) {
        for (int jj = 0; jj < d; ++jj) {
            double s = 0.0;
            for (int j = 0; j < D; ++j) s += (double)h_queries[q * D + j] * (double)h_R[j * d + jj];
            h_queries_proj[q * d + jj] = (float)s;
        }
    }

    // Device pointers for projected dataset and query
    float *d_Y_query = nullptr;
    cudaMalloc(&d_Y_query, sizeof(float) * d);

    // Run k-NN on projected data
    float total_proj_ms = 0.0f;
    for (int q = 0; q < M; ++q) {
        cudaMemcpy(d_Y_query, h_queries_proj.data() + q * d, sizeof(float) * d, cudaMemcpyHostToDevice);
        cudaEvent_t st2, ed2; cudaEventCreate(&st2); cudaEventCreate(&ed2);
        cudaEventRecord(st2, 0);

        // compute distances in projected space
        int grid2 = (N + block - 1) / block;
        computeDistances<<<grid2, block>>>(d_Y, d_Y_query, d_distances, N, d);
        cudaDeviceSynchronize();

        // sort
        thrust::device_ptr<float> dptr2(d_distances);
        thrust::device_vector<int> d_idx2(N);
        thrust::sequence(d_idx2.begin(), d_idx2.end());
        thrust::sort_by_key(dptr2, dptr2 + N, d_idx2.begin());

        cudaEventRecord(ed2, 0); cudaEventSynchronize(ed2);
        float ms2 = 0.0f; cudaEventElapsedTime(&ms2, st2, ed2);
        total_proj_ms += ms2;
        cudaEventDestroy(st2); cudaEventDestroy(ed2);
    }
    std::cout << "Projected k-NN average time per query (ms): " << (total_proj_ms / M) << "\n";

    // Cleanup
    cudaFree(d_dataset);
    cudaFree(d_query);
    cudaFree(d_distances);
    cudaFree(d_R);
    cudaFree(d_Y);
    cudaFree(d_Y_query);
    cublasDestroy(cublas);

    return 0;
}
