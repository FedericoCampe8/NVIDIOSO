//
//  cuda_bit_utilities.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 01/19/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This namespace collects a set of utilities/device functions to operate on the
//  domains/status representation on CUDA.
//  @note
//  enum class EventType 
//  {
//  	NO_EVT        = 0,
//  	SINGLETON_EVT = 1,
// 	BOUNDS_EVT    = 2,
//  	MIN_EVT       = 3,
//  	MAX_EVT       = 4,
//  	CHANGE_EVT    = 5,
//  	FAIL_EVT      = 6,
//  	OTHER_EVT     = 7
//  };
//  @note
//	Constants used to retrieve the current domain description.
//	Domain represented as:
//	| EVT | REP | LB | UB | DSZ || ... BIT ... |.
//	See system_description.h.
//


#include "cuda_bit_utilities.h"

#include <stdlib.h>
#include <stdio.h>

#if CUDAON

__device__ int
CudaBitUtils::reverse ( uint v )
{
    uint r = v;
    int s = sizeof( v ) * 8 - 1;
    for (v >>= 1; v; v >>= 1)
    {
        r <<= 1;
        r |= v & 1;
        s--;
    }
    r <<= s; // shift when v's highest bits are zero
    return r;
}//reverse

__device__ int 
CudaBitUtils::lsb ( uint v )
{
    if ( v == 0 ) return 0;
    unsigned int c = 32; // c will be the number of zero bits on the right
    v = (v & ~(v - 1));
    if (v) c--;
    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;
    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;
    return c;
}//lsb

__device__ int 
CudaBitUtils::msb ( uint v ) 
{
    if ( v == 0 ) return 0; 
    v = reverse ( v );
    unsigned int c = 32; // c will be the number of zero bits on the right
    v = (v & ~(v - 1));
    if (v) c--;
    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;
    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;

    return ((8 * sizeof(int)) - c - 1);
}//msb

__device__ bool 
CudaBitUtils::is_pwr2 ( uint v ) 
{
    return (v && !(v & (v - 1)));
}//is_pwr2

__device__ int 
CudaBitUtils::count_bit ( uint v ) 
{
    uint c;
    for (c = 0; v; c++)
    {
        v &= v - 1; // clear the least significant bit set
    }
    return c;
}//count_bit

#endif

