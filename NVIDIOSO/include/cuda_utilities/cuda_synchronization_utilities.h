//
//  cuda_synchronization_utilities.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/17/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements utilities for propagating constraints on device.
//

#ifndef __NVIDIOSO__cuda_synchronization_utilities__
#define __NVIDIOSO__cuda_synchronization_utilities__

#include "cuda_constraint_macro.h"

namespace CudaSynchUtils
{
    
#if CUDAON
    
    /**
     * Set the fields before BITs (e.g., EVT, LB, UB, DSZ),
     * consistent with the bits in BIT field.
     * @param domain_idx indexes for each variable of its starting point on the array of states
     * @param domain_states array of all the domains stored sequentially in memory
     * @note domain_idx can be used to lookup the domains corresponding to the
     *       (ids of the) variables stored in vars
     * @param domain_type the type of domain (i.e., standard, Boolean, etc.) correspondent to its size
     *        in number of integers
     * @note this function is supposed to be invoked with a number of bytes equal to the
     *       BIT + (VECTOR_MAX/8*(sizeof(int))) (or BOOL = 2) Integers.
     * @note this function is supposed to be invoked with one block per variable
     * @note domain_type_size should be either STANDARD_DOM or BOOLEAN_DOM.
     */
    __global__ void cuda_set_domains_from_bit_1b1v ( int* domain_idx, uint* domain_states, int domain_type_size = STANDARD_DOM );
    
    /**
     * Copy domains states from source_states to dest_states in parallel using 1 block per variable.
     * @param domain_idx indexes for each variable of its starting point on the array of states
     * @param source_states states to be copied
     * @param dest_states destination
     * @note domain_idx can be used to lookup the domains corresponding to the
     *       (ids of the) variables stored in vars
     * @note domain_type_size should be either STANDARD_DOM or BOOLEAN_DOM
     */
    __global__ void cuda_copy_state_1b1v ( int* domain_idx, uint* source_states, uint* dest_states, int domain_type_size = STANDARD_DOM );
    
#endif
    
}

#endif 
