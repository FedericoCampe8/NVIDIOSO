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

#if CUDAON

/**
 * Array used to store all the constraints on the device.
 * @note This is available (on device) to any class including this header file.
 * @note This array is supposed to be filled with the constraints in the model.
 *       For example, see "cuda_constraint_factory.h"
 */
class CudaConstraint;
extern __device__ CudaConstraint** g_dev_constraints;

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
     * This function is supposed to be invoked with one block per variable and the
     * constraints are propagated w.r.t. that variable.
     */
    __global__ void cuda_consistency_1b1v (  size_t * constraint_queue, int domain_type = STANDARD_DOM );
#endif
}

#endif /* defined(__NVIDIOSO__cuda_propagation_utilities__) */
