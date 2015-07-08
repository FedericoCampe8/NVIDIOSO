//
//  cuda_propagation_utilities.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 19/01/14.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//
//  This class implements utilities for propagating constraints on device.
//

#ifndef __NVIDIOSO__cuda_propagation_utilities__
#define __NVIDIOSO__cuda_propagation_utilities__

using uint = unsigned int;

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
 * Propagates the constraints in constraint_queue in parallel on device.
 * @param constraint_queue the queue of constraints to propagate.
 */
__global__ void cuda_consistency ( size_t * constraint_queue );

#endif


#endif /* defined(__NVIDIOSO__cuda_propagation_utilities__) */
