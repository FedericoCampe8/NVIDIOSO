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

// Array of global constraint
__device__ CudaConstraint** g_dev_constraints;

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
            g_dev_constraints [ constraint_queue [ i ] ]->move_status_to_shared ( shared_status, domain_type );

            g_dev_constraints [ constraint_queue [ i ] ]->consistency();
            if ( !g_dev_constraints [ constraint_queue [ i ] ]->satisfied() )
            {
                g_dev_constraints [ constraint_queue [ i ] ]->move_status_from_shared ( shared_status, domain_type );
                break;
            }

            g_dev_constraints [ constraint_queue [ i ] ]->move_status_from_shared ( shared_status, domain_type );
        }
    }
}//cuda_consistency_sequential

__global__ void
CudaPropUtils::cuda_consistency_1b1c ( size_t * constraint_queue, int domain_type )
{
    extern __shared__ uint shared_status[];
    g_dev_constraints [ constraint_queue [ blockIdx.x ] ]->move_status_to_shared ( shared_status, domain_type );

    g_dev_constraints [ constraint_queue [ blockIdx.x ] ]->consistency();
    g_dev_constraints [ constraint_queue [ blockIdx.x ] ]->satisfied();

    g_dev_constraints [ constraint_queue [ blockIdx.x ] ]->move_bit_status_from_shared ( shared_status, domain_type );
}//cuda_consistency__1b1c

__global__ void
CudaPropUtils::cuda_consistency_1b1v ( size_t * constraint_queue, int domain_type )
{
}//cuda_consistency__1b1v

#endif

