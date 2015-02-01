//
//  cuda_constraint_utility.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 19/01/15.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//
//  This class collects a set of utilities/device functions to operate on the
//  domains/status representation on CUDA.
//  @note To be consistent on naming functions, most of the following utilities (names)
//        correspond to the methods in int_variable.h.
//  @note
//  enum class EventType 
//  {
//  	NO_EVT        = 0,
//  	SINGLETON_EVT = 1,
// 	 	BOUNDS_EVT    = 2,
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


#include "cuda_constraint_utility.h"

#include <stdlib.h>
#include <stdio.h>

#if CUDAON

__device__ int 
min ( uint* domain_state )
{
	return (int) domain_state[ 2 ];
}//min

__device__ int 
max ( uint* domain_state ) 
{
	return (int) domain_state[ 3 ];
}//max

__device__ bool 
is_singleton ( uint* domain_state ) 
{
	if ( domain_state[ 2 ] == domain_state[ 3 ] ) 
	{
	 	domain_state[ 0 ] = 1;
	 	return true;
	}
	return false;
}//is_singleton

__device__ bool 
is_empty ( uint* domain_state ) 
{
	if ( domain_state[ 0 ] == 6 ) return true;
	return false;
}//is_empty

__device__ bool 
contains ( uint* domain_state, int val )
{
	/*
	 * Find the chunk and the position of the bit within the chunk.
	 * @note: chunks are stored in reversed order.
	 * 		  For example: {0, 63} is stored as
	 *        | 0 | 0 | 0 | 0 | 0 | 0 | 63...32 | 31...0 | 
	 */	 
	int chunk = val / 32;
	chunk = BIT_IDX + NUM_CHUNKS - 1 - chunk;
	return get_bit( domain_state[ chunk ], val % BITS_IN_CHUNK );
}//contains

__device__ bool  
subtract ( uint* domain_state, int val ) 
{
	// Only one pair of bounds is allowed
	int lw = domain_state [ 2 ];
	int up = domain_state [ 3 ];
	if ( val < lw || val > up ) return false;
	
	if ( ((int) domain_state [ 1 ]) == 1 ) 
	{
		// Bound representation
		if ( val == lw ) 
		{
			domain_state[ 2 ] = ++lw;
			domain_state[ 0 ] = 3;
		} 
		else if ( val == up ) {
			domain_state[ 3 ] = --up;	
			domain_state[ 0 ] = 4;
		}
		else 
		{
			return false;
		}
		
		// Switch representation if the domain is smaller than VECTOR_MAX_CUDA
		if ( (up - lw + 1) <= VECTOR_MAX_CUDA ) 
		{
			domain_state[ 1 ] = 0;
			for ( int i = 5; i < 5 + (VECTOR_MAX_CUDA / 32); i++ ) 
				domain_state[ i ] = ~0;
		}
		
		// Decrease domain size
		domain_state[ 4 ]--;
		return true;
	}  
	else 
	{
		// Bitmap representation: this should be the most common case
		
		/*
		 * Find the chunk and the position of the bit within the chunk.
		 * @note: chunks are stored in reversed order.
		 * 		  For example: {0, 63} is stored as
		 *        | 0 | 0 | 0 | 0 | 0 | 0 | 63...32 | 31...0 | 
		 */
  		int chunk = val / 32;
  		chunk = BIT_IDX + NUM_CHUNKS - 1 - chunk;
  		
  		if ( !get_bit( domain_state[ chunk ], val % BITS_IN_CHUNK ) ) return false;
  		domain_state[ chunk ] = clear_bit( domain_state[ chunk ], val % BITS_IN_CHUNK );
  		domain_state[ 4 ]--;

  		// Failed event if size is 0
  		if ( domain_state[ 4 ] == 0 ) 
  		{
  			domain_state[ 0 ] = 6;
  			return true;
  		} 
  		
  		// Singleton event if size is 1
  		if ( domain_state[ 4 ] == 1 ) 
  		{
  			if ( domain_state[ 2 ] == val ) 
  			{
  				domain_state[ 2 ] = domain_state[ 3 ];
  				domain_state[ 0 ] = 1;
  				return true;
  			}
  			else if ( domain_state[ 3 ] == val ) 
  			{
  				domain_state[ 3 ] = domain_state[ 2 ];
  				domain_state[ 0 ] = 1;
  				return true;
  			}
  		}
  		else 
  		{
  			/*
  			 * Update bounds if a lb has been increased or a up has been decreased.
  			 * Note the holes in the domain.
  			 * Note that 
  			 * get_bit( domain_state[ chunk ], val % BITS_IN_CHUNK )
  			 * corresponds to
  			 * contains ( val, domain_state )
  			 * not invoked here for efficiency reasons.
  			 */
  			 if ( domain_state[ 2 ] == val )
  			 {
  			 	// Set new lower bound
  			 	while ( true )
  			 	{
  			 		++val;
  			 		if ( get_bit( domain_state[ chunk ], val % BITS_IN_CHUNK ) ) 
  			 		{
  			 			domain_state[ 2 ] = val;
  			 			domain_state[ 0 ] = 3;
  			 			return true;
  			 		}
  			 	}
  			 }
  			 else if ( domain_state[ 3 ] == val ) 
  			 {
  			 	// Set new upper bound
  			 	while ( true )
  			 	{
  			 		--val;
  			 		if ( get_bit( domain_state[ chunk ], val % BITS_IN_CHUNK ) ) 
  			 		{
  			 			domain_state[ 3 ] = val;
  			 			domain_state[ 0 ] = 4;
  			 			return true;
  			 		}
  			 	}
  	 		}
  	 		else 
  	 		{
  	 			domain_state[ 0 ] = 5;
  	 		}	
  		}
	}
	return false;
}//subtract


#endif

