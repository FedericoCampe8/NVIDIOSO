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

	//blocchi paralleli lavorano sugli stessi domini -> raice condition
	//chiamare il kernel con un blocco per variabli e propagare i relativi vincoli 
	if (blockIdx.x == 0) {
	//gridDim.x
	int j = 0;
	for (int i = 0; i < 50; i++) {
	j = i % 30;
	printf ("VALUE %d\n",constraint_queue [ i ] );
	d_constraints_ptr [ constraint_queue [ j ] ]->consistency(); 
	d_constraints_ptr [ constraint_queue [ j ] ]->satisfied();
	//d_constraints_ptr [ constraint_queue [ i ] ]->print();
	}
	}
}//cuda_consistency


#endif

