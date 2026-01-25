#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Simple brute-force k-NN: for each query, compute distances to all dataset points on GPU,
// then use Thrust to sort distances and return top-k indices.

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
    // Parameters (can be passed as args)
    int N = 10000; // dataset size
    int D = 32;    // dimensionality
    int M = 5;     // number of queries
    int K = 5;     // neighbors

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) D = atoi(argv[2]);
    if (argc > 3) M = atoi(argv[3]);
    if (argc > 4) K = atoi(argv[4]);

    std::srand((unsigned)std::time(nullptr));

    // Host data
    std::vector<float> h_dataset(N * D);
    std::vector<float> h_queries(M * D);

    for (int i = 0; i < N * D; ++i) h_dataset[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * D; ++i) h_queries[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory
    float *d_dataset = nullptr;
    float *d_query = nullptr;
    float *d_distances = nullptr;
    cudaMalloc(&d_dataset, sizeof(float) * N * D);
    cudaMalloc(&d_query, sizeof(float) * D);
    cudaMalloc(&d_distances, sizeof(float) * N);

    cudaMemcpy(d_dataset, h_dataset.data(), sizeof(float) * N * D, cudaMemcpyHostToDevice);

    // For index handling
    thrust::device_vector<int> d_idx(N);
    thrust::sequence(d_idx.begin(), d_idx.end());

    // Work configuration
    int block = 256;
    int grid = (N + block - 1) / block;

    printf("k-NN (brute force)\nDataset size: %d, Dim: %d, Queries: %d, K: %d\n", N, D, M, K);

    for (int q = 0; q < M; ++q) {
        // copy query
        cudaMemcpy(d_query, h_queries.data() + q * D, sizeof(float) * D, cudaMemcpyHostToDevice);

        // compute distances
        computeDistances<<<grid, block>>>(d_dataset, d_query, d_distances, N, D);
        cudaDeviceSynchronize();

        // Use thrust to sort distances (device pointer) and keep indices
        thrust::device_ptr<float> dptr(d_distances);

        // reset indices for each query
        thrust::sequence(d_idx.begin(), d_idx.end());

        thrust::sort_by_key(dptr, dptr + N, d_idx.begin());

        // copy top-K indices back to host
        std::vector<int> topk(K);
        thrust::copy(d_idx.begin(), d_idx.begin() + K, topk.begin());

        // also copy top-K distances
        std::vector<float> topk_dist(K);
        thrust::copy(dptr, dptr + K, topk_dist.begin());

        // print results for this query
        printf("Query %d: Top %d neighbors:\n", q, K);
        for (int i = 0; i < K; ++i) {
            printf("  idx=%d dist=%f\n", topk[i], topk_dist[i]);
        }
    }

    // Cleanup
    cudaFree(d_dataset);
    cudaFree(d_query);
    cudaFree(d_distances);

    return 0;
}
