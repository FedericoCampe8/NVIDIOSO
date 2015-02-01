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
 * Propagates the constraints in constraint_queue in parallel on device.
 * @param constraint_queue the queue of constraints to propagate.
 */
__global__ void cuda_consistency ( size_t * constraint_queue );

#endif


#endif /* defined(__NVIDIOSO__cuda_propagation_utilities__) */
