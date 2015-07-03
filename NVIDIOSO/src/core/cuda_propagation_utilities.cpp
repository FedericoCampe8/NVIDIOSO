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
#include "cuda_cp_model.h"

using uint = unsigned int;

#if CUDAON

__global__ void
cuda_consistency ( size_t * constraint_queue )
{
	// Now everything is sequential here
	if (blockIdx.x == 0) {
		for (int i = 0; i < gridDim.x; i++) {
			//d_constraints_ptr [ constraint_queue [ i ] ]->consistency(); 
			//d_constraints_ptr [ constraint_queue [ i ] ]->satisfied();
			d_constraints_ptr [ constraint_queue [ i ] ]->print();
		}
	}
}//cuda_consistency


#endif

