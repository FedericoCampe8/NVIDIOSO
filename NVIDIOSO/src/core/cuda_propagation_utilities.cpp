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
cuda_consistency_sequential ( size_t * constraint_queue, int queue_size, int domain_type  )
{
    //extern __shared__ uint shared_status[];
     
    // Now everything is sequential here
    if (blockIdx.x == 0) 
    { 
        for (int i = 0; i < queue_size; i++) 
        {
        	//g_dev_constraints [ constraint_queue [ i ] ]->move_status_to_shared ( shared_status, domain_type );
        	
            g_dev_constraints [ constraint_queue [ i ] ]->consistency(); 
            
            //g_dev_constraints [ constraint_queue [ i ] ]->move_status_from_shared ( shared_status, domain_type );
            
            if ( !g_dev_constraints [ constraint_queue [ i ] ]->satisfied() ) break;
        }
    }
    
        
}//cuda_consistency


#endif

