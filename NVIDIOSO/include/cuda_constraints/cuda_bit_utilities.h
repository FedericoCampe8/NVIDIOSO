//
//  cuda_bit_utilities.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 01/19/15.
//  Copyright (c) 2015-2015 Federico Campeotto. All rights reserved.
//
//  This class collects a set of utilities/device functions to operate on the
//  domains/status representation on CUDA.
//

#ifndef NVIDIOSO_cuda_bit_utilities_h
#define NVIDIOSO_cuda_bit_utilities_h

#include "cuda_constraint_macro.h"

namespace CudaBitUtils
{
    
#if CUDAON
    
    /**
     * Get LSB.
     * @param unsigned int (32 bits)
     * @return LSB (counting from 0)
     */
    __device__ int lsb ( uint n );

    /**
     * Get the MSB.
     * @param unsigned int (32 bits)
     * @return MSB (counting from 0)
     */
    __device__ int msb ( uint n );

    //! Reverse an unsigned int
    __device__ int reverse ( uint n );
    
    /**
     * Checks whether a given integer (32 bits) is a power of 2, i.e., only one bit is set.
     * @param unsigned int (32 bits)
     * @return True if the given number is a power of 2, False otherwise.
     */
    __device__ bool is_pwr2 ( uint n );

    //! Counts the number of set bits in a 32 bit integer
    __device__ int count_bit ( uint n );
    
    //=================================================================
    //============  INLINE FUNCTIONS AND GENERAL UTILITIES ============
    //=================================================================

    
    /**
     * Get the bit value of the bit in position i of
     * the given unsigned int.
     * @param n usigned int to check the bit
     * @param i position of the bit to check
     * @return true iff n[i] = 1
     */
    __forceinline__ __device__ bool get_bit ( uint n, int i ) 
    {
    	if (  i < 0 || i > 31 ) return false;
    	return ( (n & (1 << i)) != 0 );
    }//get_bit
  	
    //! Clear the i^th bit and return the modified input val
    __forceinline__ __device__ uint clear_bit ( uint n, int i ) 
    {
    	if ( i < 0 || i > 31 ) return i;
    	return n &  (~(1 << i));
    }//clear_bit
 
#endif
    
}

#endif
