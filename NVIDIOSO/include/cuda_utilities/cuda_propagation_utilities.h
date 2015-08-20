//
//  cuda_propagation_utilities.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 19/01/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class implements utilities for propagating constraints on device.
//

#ifndef __NVIDIOSO__cuda_propagation_utilities__
#define __NVIDIOSO__cuda_propagation_utilities__

#include "cuda_constraint_macro.h"

#define G_DEV_CONSTRAINTS_ARRAY g_dev_constraints
#define G_DEV_GLB_CONSTRAINTS_ARRAY g_dev_glb_constraints

#if CUDAON

/**
 * Array used to store all the constraints on the device.
 * @note This is available (on device) to any class including this header file.
 * @note This array is supposed to be filled with the constraints in the model.
 *       For example, see "cuda_constraint_factory.h"
 */
class CudaConstraint;
extern __device__ CudaConstraint** g_dev_constraints;

class CudaGlobalConstraint;
extern __device__ CudaConstraint** g_dev_glb_constraints;

/**
 * Boolean array used to synchronize blocks withing a kernel.
 * When all the elements in the array are set to True,
 * all the blocks have reached the barrier and are,
 * therefore, considered as synchronized.
 */
extern __device__ bool* g_dev_synch_barrier;

#endif

namespace CudaPropUtils
{
    #if CUDAON
    /**
     * Propagates constraints in constraint_queue sequentially on device.
     * @param constraint_queue the queue of constraints to propagate.
     * @param domain_type the type of domain (i.e., standard, Boolean, etc.).
     * @note this function is created for testing and comparison purposes.
     *       It doesn't take advantage of any GPU parallelism.
     */
    __global__ void cuda_consistency_sequential ( size_t * constraint_queue, int queue_size, int domain_type = STANDARD_DOM );

    /**
     * Propagates constraints in constraint_queue in parallel on device.
     * This kernel function uses a block per constraint.
     * This kernel function should be invoked with a number of blocks equal to the number
     * of constraints in the constraint queue.
     * @param constraint_queue the queue of constraints to propagate.
     * @param domain_type the type of domain (i.e., standard, Boolean, etc.).
     */
    __global__ void cuda_consistency_1b1c (  size_t * constraint_queue, int domain_type = STANDARD_DOM );
    
    /**
     * Propagates constraints in constraint_queue in parallel on device.
     * This kernel function uses a block per K constraints.
     * This kernel function should be invoked with a number of blocks equal to the 
     * (ceil of the )number of constraints in the constraint queue, divided by block size.
     * @param constraint_queue the queue of constraints to propagate.
     * @param queue_size size of the constraint queue.
     
     * @param shared_memory_array_size size of the array of shared memory for each constraint.
     * @param domain_type the type of domain (i.e., standard, Boolean, etc.).
     */
    __global__ void cuda_consistency_1bKc (  size_t * constraint_queue, std::size_t constraint_queue_size,
 
    int shared_memory_array_size, int domain_type = STANDARD_DOM );
    
    /**
     * Propagates constraints in constraint_queue in parallel on device.
     * This function is supposed to be invoked with one block per variable and the
     * constraints are propagated w.r.t. that variable.
     * @param constraint_queue queue of constraints for each variable: this queue contains
     *        1 - Id of the constraint
     * 		  2 - Index (0, 1, 2, etc.) of the variable to consider for propagation
     * @param queue_idx indexes of the starting position for each queue of constraints for each variable
     * @param domain_type type of domain (bool, standard, mixed)
     * @aux_state pointer to an array of auxiliary states. Used for synchronization between blocks
     * @todo use a separate array for indexes of variables
     * @todo return asap if a failure is found
     */
    __global__ void cuda_consistency_1b1v (  size_t * constraint_queue, int* queue_idx, int domain_type = STANDARD_DOM, uint * aux_state = nullptr );
#endif
}

#endif /* defined(__NVIDIOSO__cuda_propagation_utilities__) */
