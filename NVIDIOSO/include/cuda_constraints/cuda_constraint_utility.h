//
//  cuda_constraint_utility.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 19/01/15.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//
//  This class collects a set of utilities/device functions to operate on the
//  domains/status representation on CUDA.
//  @note To be consistent on naming functions, most of the following utilities (names)
//        correspond to the methods in int_variable.h.
//

#ifndef NVIDIOSO_cuda_constraint_utilities_h
#define NVIDIOSO_cuda_constraint_utilities_h

using uint = unsigned int;

// This should correspond to VECTOR_MAX in globals.h
constexpr int VECTOR_MAX_CUDA = 256;

constexpr int BIT_IDX = 5;
constexpr int NUM_CHUNKS    = VECTOR_MAX_CUDA / 32;
constexpr int BITS_IN_CHUNK = 32;

/**
 * Macro for the size of a chunk in terms of bits.
	static constexpr int BITS_IN_CHUNK = sizeof( int ) * BITS_IN_BYTE;
*/
/**
 * Get index of the chunk of bits containing the bit
 * representing the value given in input.
 * @param max lower bound used to calculated the index of the bitmap
 * @return number of int used as bitmaps to represent max
 
 	static constexpr int IDX_CHUNK ( int val ) {
    	return val / ( sizeof(int) / 32 );
 	}
 */

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>

/**
 * Get the lower bound.
 * @param domain_state array representing the domain status.
 * @return an integer value representing the lower bound of the given domain.
 * @note For efficiency reasons it is better to test directly on domain_state
 * 		 without invoking this function.
 */
__device__ int min ( uint* domain_state );

/**
 * Get the upper bound.
 * @param domain_state array representing the domain status.
 * @return an integer value representing the upper bound of the given domain.
 * @note For efficiency reasons it is better to test directly on domain_state
 * 		 without invoking this function.
 */
__device__ int max ( uint* domain_state );

/**
 * Test if the domain is singleton and set SINGLETON event if it is singleton.
 * @param domain_state array representing the domain status.
 * @return true if the domain is singleton, false otherwise.
 * @note For efficiency reasons it is better to test directly on domain_state
 * 		 without invoking this function.
 */
__device__ bool is_singleton ( uint* domain_state );

/**
 * Test if the domain is empty.
 * @param domain_state array representing the domain status.
 * @return true if the domain is empty, false otherwise.
 * @note For efficiency reasons it is better to test directly on domain_state
 * 		 without invoking this function.
 */
 __device__ bool is_empty ( uint* domain_state );
 
 /**
  * Subtract a given (integer) value from a domain
  * @param domain_state array representing the domain status.
  * @param val the value to subtract from the current domain
  * @return true if succeed an element has been removed, false otherwise.
  * @note If the domain size becomes smaller than or equal to VECTOR_MAX
  * 	  this function automatically changes the internal representation of the domain
  *       to a bitmap.
  */
  __device__ bool subtract ( uint* domain_state, int val );
  
  /**
   * It checks whether the value belongs to
   * the domain or not.
   * @param domain_state array representing the domain status.
   * @param val to check whether it is in the current domain.
   * @note val is given w.r.t. the lower bound of 0.
   */
  __device__ bool contains ( uint* domain_state, int val );
 
 /////////////////////////////////////////////////////////////////////
 ////////////// INLINE FUNCTIONS AND GENERAL UTILITIES  //////////////
 /////////////////////////////////////////////////////////////////////
 
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

#endif
