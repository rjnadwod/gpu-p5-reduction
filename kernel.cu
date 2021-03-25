/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float sdata[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < size)
       sdata[t] = in[start + t];
    else
       sdata[t] = 0;

    if (start + BLOCK_SIZE + t < size)
       sdata[BLOCK_SIZE + t] = in[start + BLOCK_SIZE + t];
    else
       sdata[BLOCK_SIZE + t] = 0;

    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          sdata[t] += sdata[t+stride];
    }

    if (t == 0)
       out[blockIdx.x] = sdata[0];
}
