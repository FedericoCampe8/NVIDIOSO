//
//  cuda_propagation.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 01/19/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class provides utilities to perform constraint propagation on device.
//

#include "cuda_propagation_utilities.h"
#include "cuda_constraint.h"

using uint = unsigned int;

#if CUDAON

// Global array of base constraint
__device__ CudaConstraint** g_dev_constraints;

// Global array of global constraint
__device__ CudaConstraint** g_dev_glb_constraints;

// Array for barrier synchronization
__device__ bool* g_dev_synch_barrier;

__global__ void
CudaPropUtils::cuda_consistency_sequential ( size_t * constraint_queue, int queue_size, int domain_type  )
{
    extern __shared__ uint shared_status[];
	
    // Now everything is sequential here
    if (blockIdx.x == 0)
    {
        for (int i = 0; i < queue_size; i++)
        {
        	G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ i ] ]->move_status_to_shared ( shared_status, domain_type );
        	
            G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ i ] ]->consistency();
            if ( !G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ i ] ]->satisfied() )
            { 
            	G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ i ] ]->move_bit_status_from_shared ( shared_status, domain_type );
            	break;
            }
            G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ i ] ]->move_bit_status_from_shared ( shared_status, domain_type );
        }
    }
}//cuda_consistency_sequential

__global__ void
CudaPropUtils::cuda_consistency_1b1c ( size_t * constraint_queue, int domain_type )
{
    extern __shared__ uint shared_status[];
    G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ blockIdx.x ] ]->move_status_to_shared ( shared_status, domain_type );

    G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ blockIdx.x ] ]->consistency();
    G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ blockIdx.x ] ]->satisfied(); 

    G_DEV_CONSTRAINTS_ARRAY [ constraint_queue [ blockIdx.x ] ]->move_bit_status_from_shared ( shared_status, domain_type );
    
}//cuda_consistency__1b1c

__global__ void
CudaPropUtils::cuda_consistency_1bKc ( size_t * constraint_queue, size_t constraint_queue_size, int shared_array_size, int domain_type )
{
    extern __shared__ uint shared_status[];
	
	// Warp idx
  	int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  	int c_idx    = constraint_queue [ warp_idx ];
	
	// Local shared memory
  	uint * local_shared = &shared_status [ (threadIdx.x / WARP_SIZE) * shared_array_size ];
  
  	if ( warp_idx < constraint_queue_size )
  	{
  		G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->
  		move_status_to_shared ( local_shared, domain_type, -((threadIdx.x / WARP_SIZE) * WARP_SIZE));
  	}
  	
 	__syncthreads();
  
  	if ( ((threadIdx.x % WARP_SIZE) == 0) && (warp_idx < constraint_queue_size) )
  	{
   		G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->consistency();
    	G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->satisfied();
  	}
  
  	__syncthreads();
  
  	if ( warp_idx < constraint_queue_size)
  	{
  		G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->
  		move_bit_status_from_shared ( local_shared, domain_type, -1, nullptr, -((threadIdx.x / WARP_SIZE) * WARP_SIZE) );
  	}
    
}//cuda_consistency_1bKc

__global__ void
CudaPropUtils::cuda_consistency_1b1v ( size_t * constraint_queue, int* queue_idx, int domain_type, uint * aux_state )
{
    extern __shared__ uint shared_status[];

    int c_idx, v_idx;
    int s_con = queue_idx [ blockIdx.x ];
    int e_con = queue_idx [ blockIdx.x + 1 ];
    for ( ; s_con < e_con; s_con += 2 )
    {
    	/*
    	 * Get the current constraint to propagate w.r.t. 
    	 * the variable associated to this block
    	 */
        c_idx = constraint_queue [ s_con     ];
        v_idx = constraint_queue [ s_con + 1 ];
        
        // Copy state from global memory to shared memory to speedup operations
    	G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->move_status_to_shared ( shared_status, domain_type );

		// Perform consistency w.r.t. the variable associated to this block 
        G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->consistency ( v_idx );
        
        
        if ( !G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->satisfied() )
        {
            G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->move_bit_status_from_shared ( shared_status, domain_type, v_idx, aux_state );
            break;
        }
		
		/*
		 * Copy reduced domains back to global memory.
		 * @note DO NOT copy on the original global memory since other block 
		 *       may still be reading original global memory.
		 */
        G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->move_bit_status_from_shared ( shared_status, domain_type, v_idx, aux_state );
    }
}//cuda_consistency__1b1v

__global__ void
CudaPropUtils::cuda_consistency_1bKv ( size_t * constraint_queue, int* queue_idx, std::size_t constraint_queue_idx_size,
 int shared_array_size, int domain_type, uint * aux_state )
{
    extern __shared__ uint shared_status[];
	__shared__ bool exit_asap;
	
	if (threadIdx.x == 0 )
	{
		exit_asap = false;
	}
	__syncthreads();
	
	// Warp idx
	int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
	
	// Local shared memory
  	uint * local_shared = &shared_status [ (threadIdx.x / WARP_SIZE) * shared_array_size ];
  	

    int c_idx, v_idx;
    int s_con = queue_idx [ warp_idx % constraint_queue_idx_size     ];
    int e_con = queue_idx [ warp_idx % constraint_queue_idx_size + 1 ];

	// Loop through all constraints involving the current variable
    for ( ; s_con < e_con; s_con += 2 )
    {
    	/*
    	 * Get the current constraint to propagate w.r.t. 
    	 * the variable associated to this block
    	 */
        c_idx = constraint_queue [ s_con     ];
        v_idx = constraint_queue [ s_con + 1 ];
        		
        // Copy state from global memory to shared memory to speedup operations
		if ( warp_idx < constraint_queue_idx_size && !exit_asap )
		{
    		G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->
    		move_status_to_shared ( local_shared, domain_type, -((threadIdx.x / WARP_SIZE) * WARP_SIZE));
		}			

		if ( ((threadIdx.x % WARP_SIZE) == 0) && (warp_idx < constraint_queue_idx_size) && !exit_asap )
    	{
			// Perform consistency w.r.t. the variable associated to this block 
        	G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->consistency ( v_idx ); 
        	
        	// Satisfiability check
        	if ( !G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->satisfied() )
        	{
        		exit_asap = true;
        	}
        }
        				
		/*
		 * Copy reduced domains back to global memory.
		 * @note DO NOT copy on the original global memory since other block 
		 *       may still be reading original global memory.
		 */
		 if ( warp_idx < constraint_queue_idx_size )
		 {	
        	G_DEV_CONSTRAINTS_ARRAY [ c_idx ]->
        	move_bit_status_from_shared ( local_shared, domain_type, v_idx, aux_state, -((threadIdx.x / WARP_SIZE) * WARP_SIZE) );
        	if ( exit_asap ) break; 
        }
    }//s_con	
    
}//cuda_consistency_1bKv

#endif
