//
//  cuda_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 12/02/14.
//  Modified on 07/14/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "cuda_constraint.h"

#if CUDAON
__device__
CudaConstraint::CudaConstraint ( int n_id, int n_vars, int n_args,
                                 int* vars, int* args,
                                 int* domain_idx, uint* domain_states ) :
_unique_id   ( n_id ),
_scope_size  ( n_vars ),
_args_size   ( n_args ),
_vars        ( vars ),
_args        ( args ) {
  _status            = (uint**) malloc ( _scope_size * sizeof (uint*) );
  _temp_status       = (uint**) malloc ( _scope_size * sizeof (uint*) );
  _status_idx_lookup = (int*) malloc ( _scope_size * sizeof (int) );
  for ( int i = 0; i < _scope_size; i++ )
  {
    _status[ i ]             = &domain_states[ domain_idx[ _vars[ i ] ] ];
    _status_idx_lookup [ i ] =  domain_idx[ _vars[ i ] ];
  }//i
}//CudaConstraint

__device__
CudaConstraint::~CudaConstraint () {
  free ( _vars );
  free ( _args );
  free ( _status );
  free ( _temp_status );
  free ( _status_idx_lookup );
}//~Constraint

__device__ void
CudaConstraint::move_status_to_shared (  uint * shared_ptr, int dom_size )
{
    if ( shared_ptr == nullptr ) return;
	
    if ( blockDim.x * blockDim.y == 1 || dom_size == MIXED_DOM )
    {// One thread per block
        
        if ( threadIdx.x == 0 )
        {
            int d_size, idx = 0;
            for ( int i = 0; i < _scope_size; i++ )
            {
                _temp_status [ i ] = _status     [ i ];
                _status      [ i ] = &shared_ptr [ idx ];
                
                if ( _status [ i ][ EVT ] == BOL_EVT )
                {
                    d_size = 2;
                }
                else
                {
                    d_size = STANDARD_DOM;
                } 
                for ( int j = 0; j < d_size; j++ )
                {
                    shared_ptr [ idx++ ] = _temp_status [ i ][ j ];
            	}
            }//i 
        }
    }
    else if ( blockDim.x * blockDim.y >= _scope_size )
    {// One thread per variable
        int tid = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
        if ( tid < _scope_size )
        {
            _temp_status [ tid ] = _status [ tid ];
            if ( dom_size == STANDARD_DOM || dom_size == BOOLEAN_DOM )
            {
                _status [ tid ] = &shared_ptr [ tid * dom_size ];
                for ( int i = 0; i < dom_size; i++ )
                    shared_ptr[tid * dom_size + i] = _temp_status [ tid ][ i ]; 
            }
           
        }
    }
    else
    {// Less than one thread per variable
        int tid = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
        _temp_status [ tid ] = _status [ tid ];
        if ( dom_size == STANDARD_DOM || dom_size == BOOLEAN_DOM )
        {
            _status [ tid ] = &shared_ptr [ tid * dom_size ];
            for ( int i = 0; i < dom_size; i++ )
                shared_ptr[tid * dom_size + i] = _temp_status [ tid ][ i ];
        }

        tid += (blockDim.x * blockDim.y) + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x); 
        if ( tid < _scope_size )
        {
            _temp_status [ tid ] = _status [ tid ];
            if ( dom_size == STANDARD_DOM || dom_size == BOOLEAN_DOM )
            {
                _status [ tid ] = &shared_ptr [ tid * dom_size ];
                for ( int i = 0; i < dom_size; i++ )
                    shared_ptr[tid * dom_size + i] = _temp_status [ tid ][ i ];
            }
        }
    }
    __syncthreads();
}//move_status_to_shared

__device__ void
CudaConstraint::move_status_from_shared ( uint * shared_ptr, int dom_size, int ref )
{
    if ( shared_ptr == nullptr ) return;

    if ( blockDim.x * blockDim.y == 1 || dom_size == MIXED_DOM )
    {// One thread per block
        
        if ( threadIdx.x == 0 )
        {
            int d_size, idx = 0;
            for ( int i = 0; i < _scope_size; i++ )
            {
                if ( ref >= 0 && i != ref ) 
                {
                	if ( _status [ i ][ EVT ] == BOL_EVT )
                	{
                		idx += BOOLEAN_DOM;
                	}
                	else
                	{
                		idx += STANDARD_DOM;
                	}
                	
                	continue;
                }
                
                if ( _status [ i ][ EVT ] == BOL_EVT )
                {
                    d_size = 2;
                }
                else
                {
                    d_size = STANDARD_DOM;
                }
                
                // EVT
                if ( shared_ptr [ idx ] == FAL_EVT )
                {
                    _temp_status [ i ][ EVT ] = FAL_EVT;
                    _status [ i ] = _temp_status[ i ];
                    break;
                }
                else if ( shared_ptr [ idx ] == SNG_EVT )
                {
                    // (old == NOP_EVT ? SNG_EVT : old)
                    /*
                     * int old = atomicCAS ( &_temp_status [ i ][ EVT ], NOP_EVT, SNG_EVT );
                     * if ( old != FAL_EVT ) _temp_status [ i ][ EVT ] = SNG_EVT;
                     */
                    atomicCAS ( &_temp_status [ i ][ EVT ], NOP_EVT, SNG_EVT );
                    atomicCAS ( &_temp_status [ i ][ EVT ], BND_EVT, SNG_EVT );
                    atomicCAS ( &_temp_status [ i ][ EVT ], MIN_EVT, SNG_EVT );
                    atomicCAS ( &_temp_status [ i ][ EVT ], MAX_EVT, SNG_EVT );
                    atomicCAS ( &_temp_status [ i ][ EVT ], CHG_EVT, SNG_EVT );
                }
                else if ( d_size != BOOLEAN_DOM )
                {
                    atomicCAS ( &_temp_status [ i ][ EVT ], NOP_EVT, shared_ptr [ idx ] );
                }
                else
                {// Bool representation
                    atomicMin ( &_temp_status [ i ][ ST ], shared_ptr [ idx + ST ] );
                    idx += BOOLEAN_DOM;
                    _status [ i ] = _temp_status[ i ];
                    continue;
                }
                
                // REP
                _temp_status [ i ][ REP ] = shared_ptr [ idx + REP ];

                // LB
                atomicMax ( &_temp_status [ i ][ LB ], shared_ptr [ idx + LB ] );

                // UB
                atomicMin ( &_temp_status [ i ][ UB ], shared_ptr [ idx + UB ] );

                // DSZ
                atomicMin ( &_temp_status [ i ][ DSZ ], shared_ptr [ idx + DSZ ] );

                // BIT
                idx += BIT;
                int dom_empty = 0;
                for ( int j = BIT; j < d_size; j++ )
                {
                    dom_empty += (
                        (atomicAnd ( &_temp_status [ i ][ j ], shared_ptr [ idx++ ] )) &
                        shared_ptr [ idx-1 ] );
                }
				
				// Sanity check for race conditions
                if ( dom_empty == 0 )
                {
                    _temp_status [ i ][ EVT ] = FAL_EVT;
                }
                
                _status [ i ] = _temp_status[ i ];
            }//i
        }
    }
    else if ( blockDim.x * blockDim.y >= _scope_size )
    {// One thread per variable
        
        int tid, loop = 1;
        int num_threads = blockDim.x + blockDim.y;
        while ( (num_threads * loop) < _scope_size )
        {
            tid = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x) + ((blockDim.x * blockDim.y) * (loop - 1));
            loop++;
            
            if ( tid < _scope_size )
            {
                if ( ref >= 0 && tid != ref ) 
                {
                	_status [ tid ] = _temp_status [ tid ];
                	break;
                }
                
                if ( dom_size == STANDARD_DOM || dom_size == BOOLEAN_DOM )
                {
                    // EVT
                    if ( shared_ptr [ tid * dom_size ] == FAL_EVT )
                    {
                        _temp_status [ tid ][ EVT ] = FAL_EVT;
                    }
                    else if ( shared_ptr [ tid * dom_size ] == SNG_EVT )
                    {
                        // (old == NOP_EVT ? SNG_EVT : old)
                        int old = atomicCAS ( &_temp_status [ tid ][ EVT ], NOP_EVT, SNG_EVT );
                        if ( old != FAL_EVT )
                        {
                            _temp_status [ tid ][ EVT ] = SNG_EVT;
                        }
                    }
                    else if ( dom_size != BOOLEAN_DOM )
                    {
                        int tid_idx = tid * dom_size;
                        atomicMin ( &_temp_status [ tid ][ EVT ], shared_ptr [ tid_idx ] );
                        
                        // REP
                        _temp_status [ tid ][ REP ] = shared_ptr [ tid_idx + REP ];
                        
                        // LB
                        atomicMax ( &_temp_status [ tid ][ LB ], shared_ptr [ tid_idx + LB ] );
                        
                        // UB
                        atomicMin ( &_temp_status [ tid ][ UB ], shared_ptr [ tid_idx + UB ] );
                        
                        // DSZ
                        atomicMin ( &_temp_status [ tid ][ DSZ ], shared_ptr [ tid_idx + DSZ ] );
                        
                        // BIT
                        tid_idx += BIT;
                        for ( int j = BIT; j < dom_size; j++ )
                        {
                            atomicAnd ( &_temp_status [ tid ][ j ], shared_ptr [ tid_idx++ ] );
                        }
                    }
                    else
                    {// Bool representation
                        atomicMin ( &_temp_status [ tid ][ ST ], shared_ptr [ tid * dom_size + ST ] );
                    }
                    _status [ tid ] = _temp_status [ tid ];
                }
                else
                {
                    printf ("cuda_constraint: Error on copying stated from shared to global\n");
                }
            }
        }
    }
    __syncthreads();
}//move_status_from_shared

__device__ void
CudaConstraint::move_bit_status_from_shared ( uint * shared_ptr, int d_size, int ref, uint* extern_status )
{
    if ( shared_ptr == nullptr ) return;
    if ( blockDim.x * blockDim.y == 1 || d_size == MIXED_DOM )
    {// One thread per block
        if ( threadIdx.x == 0 )
        {
        	bool to_extern = ( extern_status != nullptr );
            int idx = 0, ext_idx;
            for ( int i = 0; i < _scope_size; i++ )
            {
            	ext_idx = _status_idx_lookup [ i ];
                if ( ref >= 0 && i != ref ) 
                {
                	if ( d_size == BOOLEAN_DOM )
                	{
                		idx += BOOLEAN_DOM;
                	}
                	else
                	{
                		idx += STANDARD_DOM;
                	}
                	_status [ i ] = _temp_status[ i ];
                	continue;
                }
                
                // EVT
                if ( shared_ptr [ idx ] == FAL_EVT )
                {
                	if ( !to_extern )
                	{ 
                		_temp_status [ i ][ EVT ] = FAL_EVT;	
                    }
                    else
                    {
                    	//extern_status[map[i] + EVT] = FAL_EVT
                    	extern_status[ ext_idx + EVT ] = FAL_EVT;
                    }
                }
                else if ( d_size == BOOLEAN_DOM )
                {// Bool representation
                	if ( !to_extern  )
                	{
                    	atomicMin ( &_temp_status [ i ][ ST ], shared_ptr [ idx + ST ] );
                    }
                    else
                    {
                    	atomicMin ( &extern_status [ ext_idx +  ST ], shared_ptr [ idx + ST ] );
                    }
                    idx += BOOLEAN_DOM;
                    
                    _status [ i ] = _temp_status[ i ];
                    continue;
                }
                
                idx += BIT;
                for ( int j = BIT; j < d_size; j++ )
                {
                	if ( !to_extern )
                	{
                    	atomicAnd ( &_temp_status [ i ][ j ], shared_ptr [ idx++ ] );
                    }
                    else
                    {
                    	atomicAnd ( &extern_status [ ext_idx + j ], shared_ptr [ idx++ ] );
                    }
                }
                
                // Copy back the original pointers
                _status [ i ] = _temp_status[ i ];
            }
        }
    }
    else if ( blockDim.x * blockDim.y >= _scope_size )
    {// One thread per variable

        int tid, loop = 1;
        bool to_extern = ( extern_status != nullptr );
        int num_threads = blockDim.x + blockDim.y;
        while ( (num_threads * loop) < _scope_size )
        {
            tid = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x) + ((blockDim.x * blockDim.y) * (loop - 1));
            ++loop;
            
            if ( tid < _scope_size )
            {
                if ( ref >= 0 && tid != ref ) 
                {	
                	_status [ tid ] = _temp_status [ tid ]; 
                	break;
                }
                
                if ( d_size == STANDARD_DOM || d_size == BOOLEAN_DOM )
                {
                    // EVT
                    if ( shared_ptr [ tid * d_size ] == FAL_EVT )
                    {
                    	if ( !to_extern )
                		{
                        	_temp_status [ tid ][ EVT ] = FAL_EVT;
                        }
                        else
                        {
                        	extern_status [ _status_idx_lookup [ tid ] + EVT ] = FAL_EVT;
                        }
                    }
                    else if ( d_size == BOOLEAN_DOM )
                    {//Bool representation
                    	if ( !to_extern )
                		{
                        	atomicMin ( &_temp_status [ tid ][ ST ], shared_ptr [ tid * d_size + ST ] );
                        }
                        else
                        {
                        	atomicMin ( &extern_status [ _status_idx_lookup [ tid ] + ST ], shared_ptr [ tid * d_size + ST ] );
                        }
                    }
                    else
                    {
                        int tid_idx = tid * d_size;
                        tid_idx += BIT;
                        for ( int j = BIT; j < d_size; j++ )
                        {
                        	if ( !to_extern )
                			{
                        		atomicAnd ( &_temp_status [ tid ][ j ], shared_ptr [ tid_idx++ ] );
                        	}
                        	else
                        	{
                        		atomicAnd ( &extern_status [ _status_idx_lookup [ tid ] + j ], shared_ptr [ tid_idx++ ] );
                        	}
                        }
                    }
                    _status [ tid ] = _temp_status [ tid ];
                }
            }
        }
    }
}//move_bit_status_from_shared

__device__ bool 
CudaConstraint::all_ground () const
{
    for ( int i = 0; i < _scope_size; i++ )
    {
        if ( _status[ i ][ EVT ] != SNG_EVT )
        {
            return false;
        }
    }
    return true;
}//all_ground

__device__ bool
CudaConstraint::only_one_not_ground () const
{
    int not_grd = 0;
    for ( int i = 0; i < _scope_size; i++ )
    {
        if ( _status[ i ][ EVT ] != SNG_EVT )
        {
            not_grd++;
            if ( not_grd > 1 ) 
                return false;
        }
    }
    return true;
}//only_one_not_ground

__device__ bool
CudaConstraint::is_singleton ( int var ) const
{
    return (_status[ var ][ EVT ] == SNG_EVT); 
}//is_singleton

__device__ bool
CudaConstraint::is_ground ( int var ) const
{
    return _status[ var ][ EVT ] == SNG_EVT;
}//is_ground

__device__ int
CudaConstraint::get_not_ground () const
{
    for ( int i = 0; i < _scope_size; i++ )
        if ( _status[ i ][ EVT ] != SNG_EVT )
            return i;
    return -1;
}//get_not_ground

__device__ void
CudaConstraint::subtract ( int var, int val, int ref )
{
    /*
     * Three representations of domains to consider:
     * 1) Bitmap for non Boolean vars
     * 2) Bounds for non Boolean vars
     * 3) Bits for Boolean vars
     * @note this function is also in charge of setting
     *       singleton, empty/failed domain in EVT field.
     */

    /*
     * Return if there is a reference on the variable
     * to propagate and the reference doesn't match
     * with the current variable var given in input.
     */
    if ( ref >= 0 && ref != var ) return;
    
    // If already failed event, return
    if ( _status[ var ][ EVT ] == FAL_EVT ) return; 
    
    // Boolean domain
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        if ( val < 0 || val > 1 ) return;
        
        int b_val = _status[ var ][ ST ];
        if ( b_val == val )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _status[ var ][ EVT ] = FAL_EVT;
        }
        else
        {// b_val is BOL_U (undef) -> remove one val and set singleton
            _status[ var ][ ST ]   = val;
            _status[ var ][ EVT ]  = SNG_EVT;
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _status [ var ][ LB ];
        int upper_bound = _status [ var ][ UB ];
        
        if ( val < lower_bound || val > upper_bound ) return;
        if ( _status [ var ][ REP ] == 0 )
        {// Bitmap representation for domain
            /*
             * Find the chunk and the position of the bit within the chunk.
             * @note: chunks are stored in reversed/LSB order.
             *   For example: {0, 63} is stored as
             *        | 0 | 0 | 0 | 0 | 0 | 0 | 63...32 | 31...0 |
             */
            int chunk = val / BITS_IN_CHUNK;
            chunk = BIT + NUM_CHUNKS - 1 - chunk;
            uint val_chunk = _status[ var ][ chunk ];
            uint val_clear = val % BITS_IN_CHUNK;
            
            // Check if the element val is in the domain, if not return
            if ( !((val_chunk & ( 1 << val_clear )) != 0) ) return;
            
            // If val belongs to the domain, subtract it
            _status[ var ][ chunk ] = val_chunk & (~(1 << val_clear ));
            
            // Make the size consistent (avoid race conditions)
            int domain_size = 0;
            for ( int i = BIT; i < BIT + NUM_CHUNKS; i++ )
            {
            	if ( _status[ var ][ i ] == 0 ) continue;
            	domain_size += num_1bit ( _status[ var ][ i ] );
            }
            
            // Empty domain: fail event
            if ( domain_size <= 0 )
            {// Failed event
                _status[ var ][ EVT ] = FAL_EVT;
                return;
            }
			
			// Singleton event
            if ( domain_size == 1 )
            {
            	_status[ var ][ DSZ ] = 1;
            	
                // Lower bound increased: val was the lower bound
                if ( lower_bound == val )
                {
                    _status[ var ][ LB ]  = upper_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                // Upper bound decreased: val was the upper bound
                _status[ var ][ UB ]  = lower_bound;
                _status[ var ][ EVT ] = SNG_EVT;
                return;
            }
            
            // At least two elements (after subtracting val)
            if ( lower_bound == val )
            {// Lower bound increased
            
            	while ( lower_bound <= upper_bound ) 
            	{// Find new lower bound
                    ++lower_bound;
                    if ( contains ( var, lower_bound ) ) 
                    {
                        _status[ var ][ LB ]  = lower_bound;
                        break;
                    }
                }
                _status[ var ][ EVT ] = MIN_EVT;
            }
            else if ( upper_bound == val )
            {// Upper bound decreased
            
            	while ( upper_bound >= lower_bound ) 
            	{// Find new upper bound
                    --upper_bound;
                    if ( contains ( var, upper_bound ) ) 
                    {
                        _status[ var ][ UB ]  = upper_bound;
                        break;
                    }
                }
            	_status[ var ][ EVT ] = MAX_EVT;
            }
            else
            {
            	_status[ var ][ EVT ] = CHG_EVT;
            }
            
            // Set new domain size
            _status[ var ][ DSZ ] = domain_size;
        }
        else
        {// Pair of bounds
            if ( val > lower_bound && val < upper_bound ) return;
            if ( lower_bound == val )
            {
                if ( upper_bound == val )
                {
                    _status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( upper_bound == val+1 )
                {
                    _status[ var ][ LB ]  = upper_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                _status[ var ][ LB ]  = lower_bound + 1;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            else if ( upper_bound == val )
            {
                if ( lower_bound == val )
                {
                    _status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( lower_bound == val-1 )
                {
                    _status[ var ][ UB ]  = lower_bound;
                    _status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                _status[ var ][ UB ]  = upper_bound - 1;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
        }
    }
}//subtract

__device__ int
CudaConstraint::get_min ( int var ) const
{
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val = _status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_F;
        }
        return val;
    }

    // Standard domain representation
    return _status[ var ][ LB ];
}//get_min

__device__ int
CudaConstraint::get_max ( int var ) const
{
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val = _status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_T;
        }
        return val;
    }

    // Standard domain representation
    return _status[ var ][ UB ];
}//get_max

__device__ int
CudaConstraint::get_sum_ground () const
{
    int product = 0;
    for ( int idx = 0; idx < NUM_VARS; idx++ )
    {
        if ( is_singleton ( idx ) )
        {
            product += ARGS[idx] * get_min ( idx );
        }
    }
    return product;
}//get_sum_ground

__device__ bool
CudaConstraint::is_empty ( int var ) const
{
    return ( _status[ var ][ EVT ] == FAL_EVT );
}//is_empty

__device__ void
CudaConstraint::shrink ( int var, int smin, int smax, int ref )
{
    /*
     * Three representations of domains to consider:
     * 1) Bitmap for non Boolean vars
     * 2) Bounds for non Boolean vars
     * 3) Bits for Boolean vars
     * @note this function is also in charge of setting
     *       singleton, empty/failed domain in EVT field.
     */

    /*
     * Return if there is a reference on the variable
     * to propagate and the reference doesn't match
     * with the current variable var given in input.
     */
    if ( ref >= 0 && ref != var ) return;
    
    // Boolean domain
    if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        if ( smin < 0 || smin > 1 || smax < 0 || smax > 1 || smax < smin ) return;
        if ( smin + smax == 1 )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _status[ var ][ EVT ] = FAL_EVT;
        }
        if ( smin == smax )
        {
            subtract ( var, smin );
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _status [ var ][ LB ];
        int upper_bound = _status [ var ][ UB ];
        if ( smin <= lower_bound && smax >= upper_bound ) return;
        if ( smin > smax )
        {
            _status[ var ][ EVT ] = FAL_EVT;
            return;
        }
        
        smin = (smin > lower_bound) ? smin : lower_bound;
        smax = (smax < upper_bound) ? smax : upper_bound;
        
        if ( smin == smax )
        {
            _status[ var ][ DSZ ] = 1;
            _status[ var ][ LB ]  = smin;
            _status[ var ][ UB ]  = smin;
            _status[ var ][ EVT ] = SNG_EVT;
            return;
        }
        
        if ( _status [ var ][ REP ] == 0 )
        {// Bitmap representation for domain
        
            int chunk_min = smin / BITS_IN_CHUNK;
            chunk_min = BIT + NUM_CHUNKS - 1 - chunk_min;
            int chunk_max = smax / BITS_IN_CHUNK;
            chunk_max = BIT + NUM_CHUNKS - 1 - chunk_max;
            for ( int i = BIT; i < chunk_min; i++ )                  _status [ var ][ i ] = 0;
            for ( int i = BIT + NUM_CHUNKS - 1; i > chunk_max; i-- ) _status [ var ][ i ] = 0;
            clear_bits_i_through_0   ( _status [ var ][ chunk_min ], (smin % BITS_IN_CHUNK) - 1 );
            clear_bits_MSB_through_i ( _status [ var ][ chunk_max ], (smax % BITS_IN_CHUNK) + 1 );

            int num_bits = 0;
            for ( int i = chunk_min; i <= chunk_max; i++ )
                num_bits += num_1bit ( (uint) _status [ var ][ i ] );

            if ( num_bits == 0 )
            {
                _status[ var ][ EVT ] = FAL_EVT;
                return;
            }
            if ( num_bits == 1 )
            {
                _status[ var ][ DSZ ] = 1;
                _status[ var ][ LB ]  = smin;
                _status[ var ][ UB ]  = smin;
                _status[ var ][ EVT ] = SNG_EVT;
                return;
            }
            
            _status[ var ][ DSZ ] = num_bits;
            lower_bound = (smin < lower_bound) ? lower_bound : smin;
            upper_bound = (smax > upper_bound) ? upper_bound : smax;
            while ( true ) 
            {
      			if ( contains ( var, lower_bound ) ) 
      			{
        			_status[ var ][ LB ]  = lower_bound;
					break;
  				}
  				lower_bound++;
    		}
    		while ( true ) 
            {
      			if ( contains ( var, upper_bound ) ) 
      			{
    				_status[ var ][ UB ]  = upper_bound;
					break;
				}
				upper_bound--;
			}
            
            _status[ var ][ LB ]  = lower_bound;
            _status[ var ][ UB ]  = upper_bound;
            _status[ var ][ EVT ] = CHG_EVT;
        }
        else
        {// Pair of bounds
        
            if ( smin > lower_bound && smax > upper_bound )
            {
                int cnt = smin - lower_bound;
                _status[ var ][ LB ] = smin;
                _status[ var ][ DSZ ] -= cnt;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            if ( smin < lower_bound && smax < upper_bound )
            {
                int cnt = upper_bound - smax;
                _status[ var ][ UB ] = smax;
                _status[ var ][ DSZ ] -= cnt;
                _status[ var ][ EVT ] = BND_EVT;
                return;
            }
            int cnt = (smin - lower_bound) + (upper_bound - smax);
            _status[ var ][ LB ] = smin;
            _status[ var ][ UB ] = smax;
            _status[ var ][ DSZ ] -= cnt;
            _status[ var ][ EVT ] = BND_EVT;
        }
    }
}//shrink

__device__ bool 
CudaConstraint::contains ( int var, int val )
{
	if ( _status[ var ][ EVT ] == BOL_EVT )
    {
        int val_in = _status[ var ][ ST ];
        if ( val_in == BOL_U )
        {
            return (val >= 0 && val <= 1);
        }
        return val == val_in;
    }
    
    int chunk = val / BITS_IN_CHUNK;
    chunk = BIT + NUM_CHUNKS - 1 - chunk;
	return ( (_status [ var ][ chunk ] & (1 << (val % BITS_IN_CHUNK))) != 0 );
}//contains

__device__ void
CudaConstraint::clear_bits_i_through_0 ( uint& val, int i )
{
    int mask = ~((1 << (i+1)) - 1);
    val = val & mask;
}//clear_bits_i_through_0

__device__ void
CudaConstraint::clear_bits_MSB_through_i ( uint& val, int i )
{
    val = val & ( ( 1 << i ) - 1 );
}//clear_bits_MSB_through_i

__device__ int
CudaConstraint::num_1bit ( uint n )
{
    int c = 0;
    for ( c = 0; n; c++ )
        n &= n - 1;
    return c;
}//num_1bit

__device__ size_t
CudaConstraint::get_unique_id () const
{
  return _unique_id;
}//get_unique_id

__device__ size_t
CudaConstraint::get_scope_size () const
{
  return _scope_size;
}//get_scope_size

__device__ size_t
CudaConstraint::get_arguments_size () const
{
  return _args_size;
}//get_arguments_size

#endif
