//
//  cuda_synchronization_utilities.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/17/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class provides utilities to synchronize domains on device.
//

#include <stdio.h>
#include "cuda_synchronization_utilities.h"
#include "cuda_bit_utilities.h"

#if CUDAON

__global__ void
CudaSynchUtils::cuda_set_domains_from_bit_1b1v ( int* domain_idx, uint* domain_states, int domain_type_size  )
{
    // Shared memory for the variable's domain
    extern __shared__ uint shared_status[];
    
    // Move to shared
    if ( blockDim.x >= domain_type_size )
    {
        if ( threadIdx.x < domain_type_size )
        {
            shared_status[ threadIdx.x ] = domain_states[ domain_idx[ blockIdx.x ] + threadIdx.x ];
        }
    }
    else 
    {// One thread here
        if ( threadIdx.x == 0 )
        {
            uint * addr = &domain_states[ domain_idx[ blockIdx.x ] ];
            for ( int i = 0; i < domain_type_size; i++ )
            {
                shared_status[ i ] = addr[ i ];
            }
        }
    }
    __syncthreads();
    
    // Perform consistency on domains w.r.t. bit represetations
    //! @todo perform consistency in parallel
    if ( threadIdx.x == 0 && shared_status[ EVT ] != FAL_EVT )
    {
        if ( domain_type_size == BOOLEAN_DOM )
        {
            if ( shared_status [ ST ] != BOL_U )
            {
                shared_status [ EVT ] = SNG_EVT;
            }
            else
            {
                shared_status [ EVT ] = NOP_EVT;
            }
        }
        else
        {
            int lower_bnd = -1, upper_bnd = -1, msb;
            int dom_size = 0, idx = BIT;
            for ( ; idx < domain_type_size; ++idx )
            {
                if ( shared_status[ idx ] )
                {
                    dom_size += CudaBitUtils::count_bit( shared_status[ idx ] );
                    if ( lower_bnd == -1 )
                    {
                        lower_bnd = CudaBitUtils::lsb ( shared_status[ idx ] ) + (32 * (domain_type_size - idx - 1));
                    }
                    msb = CudaBitUtils::msb ( shared_status[ idx ] ) + (32 * (domain_type_size - idx - 1));
                    if ( msb > upper_bnd )
                        upper_bnd = msb;
                }
            }
            
            shared_status [ DSZ ] = dom_size;
            shared_status [ LB  ] = lower_bnd; 
            shared_status [ UB  ] = upper_bnd;
            if ( dom_size <= 0 )
            {
                shared_status [ EVT ] = FAL_EVT;
            }
            else if ( dom_size == 1 )
            {
                shared_status [ EVT ] = SNG_EVT;
            }
            else
            {
                int prev_sz = domain_states[ domain_idx[ blockIdx.x ] + DSZ ];
                if ( prev_sz != dom_size )
                {
                    int prev_lb = domain_states[ domain_idx[ blockIdx.x ] + LB ];
                    int prev_ub = domain_states[ domain_idx[ blockIdx.x ] + UB ];
                    if ( prev_lb != lower_bnd || prev_ub != upper_bnd )
                    {
                        shared_status [ EVT ] = BND_EVT;
                    }
                    else
                    {
                        shared_status [ EVT ] = CHG_EVT;
                    }
                }
            }
        }
    }
    
    // Move from shared
    if ( blockDim.x >= domain_type_size )
    {
        if ( threadIdx.x < domain_type_size )
        {
            domain_states[ domain_idx[ blockIdx.x ] + threadIdx.x ] = shared_status[ threadIdx.x ];
        }
    }
    else
    {// One thread here
        if ( threadIdx.x == 0 )
        {
            uint * addr = &domain_states[ domain_idx[ blockIdx.x ] ];
            for ( int i = 0; i < domain_type_size; i++ )
            {
                addr[ i ] = shared_status[ i ];
            }
        }
    }
}//cuda_set_domains_from_bit_1b1v

#endif

