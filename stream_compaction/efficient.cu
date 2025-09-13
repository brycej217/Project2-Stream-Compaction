#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <cassert>

#define BLOCKSIZE 128 // ensure blocksize is a power of 2 for this implementation

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scan(int n, int ilog2, int* d_data, int* d_blockSums)
        {
            int local = threadIdx.x;
            int global = (blockIdx.x * blockDim.x) + threadIdx.x;
            int blockLim = (blockIdx.x + 1) * blockDim.x - 1;

            if (global >= n) return;

            // running per block
            // upsweep
            for (int d = 0; d < ilog2; d++)
            {
                if (local % static_cast<int>(pow(2, d + 1)) == 0)
                {
                    if ((global + static_cast<int>(pow(2, d + 1)) - 1) <= blockLim)
                    {
                        d_data[global + static_cast<int>(pow(2, d + 1)) - 1] += d_data[global + static_cast<int>(pow(2, d)) - 1];
                    }
                }
                __syncthreads();
            }

            d_blockSums[blockIdx.x] = d_data[blockLim]; // store before downsweeping
            
            // downsweep
            if (global == blockLim) d_data[global] = 0;

            for (int d = ilog2 - 1; d >= 0; d--)
            {
                if (local % static_cast<int>(pow(2, d + 1)) == 0)
                {
                    if ((global + static_cast<int>(pow(2, d + 1)) - 1) <= blockLim)
                    {
                        int t = d_data[global + static_cast<int>(pow(2, d)) - 1];
                        d_data[global + static_cast<int>(pow(2, d)) - 1] = d_data[global + static_cast<int>(pow(2, d + 1)) - 1];
                        d_data[global + static_cast<int>(pow(2, d + 1)) - 1] += t;
                    }
                }
                __syncthreads();
            }
        }

        __global__ void scanBlock(int n, int ilog2, int* d_data)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            // upsweep
            for (int d = 0; d < ilog2; d++)
            {
                if (k % static_cast<int>(pow(2, d + 1)) == 0)
                {
                    d_data[k + static_cast<int>(pow(2, d + 1)) - 1] += d_data[k + static_cast<int>(pow(2, d)) - 1];
                }
                __syncthreads();
            }

            // downsweep
            if (k == n - 1) d_data[k] = 0;

            for (int d = ilog2 - 1; d >= 0; d--)
            {
                if (k % static_cast<int>(pow(2, d + 1)) == 0)
                {
                    int t = d_data[k + static_cast<int>(pow(2, d)) - 1];
                    d_data[k + static_cast<int>(pow(2, d)) - 1] = d_data[k + static_cast<int>(pow(2, d + 1)) - 1];
                    d_data[k + static_cast<int>(pow(2, d + 1)) - 1] += t;
                }
                __syncthreads();
            }
        }

        __global__ void add(int n, int* d_data, int* d_blockSums)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            d_data[k] += d_blockSums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            // declare new memory to use in scan
            int* h_data;
            int* d_data;

            // create host visible memory scaled to power of 2
            int ilog2 = static_cast<int>(ilog2ceil(n));
            int size = static_cast<int>(pow(2, ilog2));

            h_data = (int*)malloc(size * sizeof(int));
            
            // fill with zeros
            for (int i = 0; i < size; i++)
            {
                h_data[i] = 0;
            }

            // create device memory scaled to power of 2
            cudaMalloc((void**)&d_data, size * sizeof(int));
            cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyHostToDevice); // copy values into device memory

            // divide array into blocks
            int numBlocks = (size + BLOCKSIZE - 1) / BLOCKSIZE;

            // declare block sum array to support scans of arbitrary length
            int* d_blockSums;
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));

            scan<<<numBlocks, BLOCKSIZE>>>(size, ilog2ceil(BLOCKSIZE), d_data, d_blockSums);

            assert(numBlocks <= 1024); // if numBlocks is greater than 1024, we cannot operate on it

            scanBlock<<<1, numBlocks>>>(numBlocks, ilog2ceil(numBlocks), d_blockSums);

            add<<<numBlocks, BLOCKSIZE>>>(size, d_data, d_blockSums);

            cudaMemcpy(odata, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();


            timer().endGpuTimer();
            return -1;
        }
    }
}
