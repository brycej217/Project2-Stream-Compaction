#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();

            odata[0] = 0;
            for (int k = 1; k < n; k++)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
            }

            timer().endCpuTimer();
        }


        /**
         * CPU scan without timer to avoid exceptions.
         */
        void scanUntimed(int n, int* odata, const int* idata)
        {
            odata[0] = 0;
            for (int k = 1; k < n; k++)
            {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            
            int count = 0;

            timer().startCpuTimer();
            
            for (int k = 0; k < n; k++)
            {
                if (idata[k] != 0)
                {
                    odata[count] = idata[k];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* tdata = new int[n];
            int* sdata = new int[n];
            int count = 0;

            timer().startCpuTimer();
            for (int k = 0; k < n; k++)
            {
                if (idata[k] == 0)
                {
                    tdata[k] = 0;
                }
                else
                {
                    tdata[k] = 1;
                }
            }

            scanUntimed(n, sdata, tdata);

            for (int k = 0; k < n; k++)
            {
                if (tdata[k] == 1)
                {
                    odata[sdata[k]] = idata[k];
                    count++;
                }
            }

            timer().endCpuTimer();
            return count;
        }
    }
}
