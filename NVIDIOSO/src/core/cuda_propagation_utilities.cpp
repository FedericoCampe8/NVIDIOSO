//
//  cuda_simple_constraint_store.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 19/01/14.
//  Copyright (c) 2015 ___UDNMSU___. All rights reserved.
//
//  This class implements a (simple) constraint store.
//

#include "cuda_propagation_utilities.h"
#include "cuda_constraint.h"

using uint = unsigned int;

#if CUDAON
// Array of global constraint 
__device__ CudaConstraint** g_dev_constraints;

__global__ void
cuda_consistency ( size_t * constraint_queue )
{
	// Now everything is sequential here
	if (blockIdx.x == 0) 
	{ 
		for (int i = 0; i < gridDim.x; i++) 
		{
			g_dev_constraints [ constraint_queue [ i ] ]->consistency(); 
			g_dev_constraints [ constraint_queue [ i ] ]->satisfied();
			//g_dev_constraints [ constraint_queue [ i ] ]->print();
		}
	}
}//cuda_consistency


#endif

