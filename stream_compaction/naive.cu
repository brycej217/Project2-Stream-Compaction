#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define BLOCKSIZE 1024

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void scan(int n, int d, int* d_odata, int* d_idata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            if (k >= 1 << (d - 1))
            {
                d_odata[k] = d_idata[k - (1 << (d - 1))] + d_idata[k];
            }
            else
            {
                d_odata[k] = d_idata[k];
            }
        }

        __global__ void rightShift(int n, int* d_odata, int* d_idata)
        {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) return;

            d_odata[k] = (k > 0) ? d_idata[k - 1] : 0;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numBlocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;

            int* d_odata;
            int* d_idata;

            // create device memory to use in kernel
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            cudaMalloc((void**)&d_idata, n * sizeof(int));

            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // ping-pong buffers
            int currO = 0;
            int currI = 1;
            int* d_data[2]; // pointer array for ping ponging
            d_data[0] = d_odata;
            d_data[1] = d_idata;

            timer().startGpuTimer();
            

            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                scan<<<numBlocks, BLOCKSIZE>>>(n, d, d_data[currO], d_data[currI]);
                currO = (currO + 1) % 2;
                currI = (currI + 1) % 2;
            }
            rightShift<<<numBlocks, BLOCKSIZE>>>(n, d_data[currO], d_data[currI]);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, d_data[currO], n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
