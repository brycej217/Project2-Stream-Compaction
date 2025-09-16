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

        int* d_garbage;

        __global__ void scan(int n, int ilog2, int* d_data, int* d_blockSums)
        {
            int local = threadIdx.x;
            int global = (blockIdx.x * blockDim.x) + threadIdx.x;
            int blockEnd = min(n - 1, (blockIdx.x + 1) * blockDim.x - 1);

            if (global >= n) return;

            // running per block
            // upsweep
            for (int d = 0; d < ilog2; d++)
            {
                int stride = 1 << (d + 1); // 2, 4, 8,...
                int prevStride = 1 << d; // 1, 2, 4,...

                if (global + stride - 1 <= blockEnd)
                {
                    if (local % stride == 0)
                    {
                        d_data[global + stride - 1] += d_data[global + prevStride - 1];
                    }
                }
                __syncthreads();
            }

            d_blockSums[blockIdx.x] = d_data[blockEnd]; // store before downsweeping

            // downsweep
            if (global == blockEnd) d_data[global] = 0;

            for (int d = ilog2 - 1; d >= 0; d--)
            {
                int stride = 1 << (d + 1); // 2, 4, 8,...
                int prevStride = 1 << d; // 1, 2, 4,...

                if (global + stride - 1 <= blockEnd)
                {
                    if (local % stride == 0)
                    {
                        int t = d_data[global + prevStride - 1]; // store left child data
                        d_data[global + prevStride - 1] = d_data[global + stride - 1];
                        d_data[global + stride - 1] += t; // add left child to right child
                    }
                }
                __syncthreads();
            }
        }

        __global__ void add(int n, int* d_data, int* d_adder)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            d_data[k] += d_adder[blockIdx.x];
        }

        // helper function for recursive scan -- parameters: array length and data device memory pointer
        void scanRecursive(int n, int* data)
        {
            int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;

            int* d_blockSums;
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));

            scan << <numBlocks, BLOCKSIZE >> > (n, ilog2ceil(BLOCKSIZE), data, d_blockSums);

            if (numBlocks >= BLOCKSIZE)
            {
                scanRecursive(numBlocks, d_blockSums);
            }
            else
            {
                scan<< <1, BLOCKSIZE >> > (numBlocks, ilog2ceil(BLOCKSIZE), d_blockSums, d_garbage);
            }
            add << <numBlocks, BLOCKSIZE >> > (n, data, d_blockSums);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // create host visible memory scaled to power of 2
            int ilog2 = static_cast<int>(ilog2ceil(n));
            int size = static_cast<int>(pow(2, ilog2));

            int* d_odata;
            int* d_blockSums;
            cudaMalloc((void**)&d_odata, size * sizeof(int));
            cudaMemcpy(d_odata, idata, size * sizeof(int), cudaMemcpyHostToDevice); // copy values into device memory
            cudaMalloc((void**)&d_garbage, size * sizeof(int));

            // divide array into blocks
            int numBlocks = (size + BLOCKSIZE - 1) / BLOCKSIZE;
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));

            timer().startGpuTimer();
            scan<<<numBlocks, BLOCKSIZE>>>(size, ilog2ceil(BLOCKSIZE), d_odata, d_blockSums);

            scanRecursive(numBlocks, d_blockSums);

            add << <numBlocks, BLOCKSIZE >> > (n, d_odata, d_blockSums);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, size * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
        * Performs prefix-sum (aka scan) on idata, storing the result into odata, WITH data arrays being device handles
        */
        void scanGPU(int n, int* odata, const int* idata, int numBlocks) {
            // declare block sum array to support scans of arbitrary length
            int* d_blockSums;
            int* d_garbage;
            cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(int));
            cudaMalloc((void**)&d_garbage, n * sizeof(int));
            cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            scan << <numBlocks, BLOCKSIZE >> > (n, ilog2ceil(BLOCKSIZE), odata, d_blockSums);

            //assert(numBlocks <= 1024); // if numBlocks is greater than 1024, we cannot operate on it

            scan<< <1, numBlocks >> > (numBlocks, ilog2ceil(numBlocks), d_blockSums, d_garbage);

            add << <numBlocks, BLOCKSIZE >> > (n, odata, d_blockSums);
        }

        __global__ void temp(int n, int badVal, int* odata, int* idata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            if (idata[k] == badVal)
            {
                odata[k] = 0;
            }
            else
            {
                odata[k] = 1;
            }
        }

        __global__ void scatter(int n, int* odata, int* tdata, int* sdata, int* idata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            if (tdata[k] == 1)
            {
                odata[sdata[k]] = idata[k];
            }
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
        int compact(int n, int* odata, const int* idata) {

            // create memory scaled to power of 2
            int ilog2 = static_cast<int>(ilog2ceil(n));
            int size = static_cast<int>(pow(2, ilog2));

            int* d_odata;
            int* d_idata;
            int* d_tdata;
            int* d_sdata;
            cudaMalloc((void**)&d_odata, size * sizeof(int));
            cudaMalloc((void**)&d_idata, size * sizeof(int));
            cudaMalloc((void**)&d_tdata, size * sizeof(int));
            cudaMalloc((void**)&d_sdata, size * sizeof(int));

            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int numBlocks = (size + BLOCKSIZE - 1) / BLOCKSIZE;


            timer().startGpuTimer();
            temp << <numBlocks, BLOCKSIZE >> > (n, 0, d_tdata, d_idata);

            scanGPU(size, d_sdata, d_tdata, numBlocks);

            scatter << <numBlocks, BLOCKSIZE >> > (n, d_odata, d_tdata, d_sdata, d_idata);

            timer().endGpuTimer();

            // get count
            int s = 0;
            int t = 0;
            cudaMemcpy(&s, d_sdata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&t, d_tdata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int count = s + t;

            cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            return count;
        }
    }
}