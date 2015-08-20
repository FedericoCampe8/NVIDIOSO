//
//  cuda_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 12/02/14.
//  Modified on 07/14/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_constraint.h"

#if CUDAON

__device__
CudaConstraint::CudaConstraint ( int n_id, int n_vars, int n_args,
                                 int* vars, int* args,
                                 int* domain_idx, uint* domain_states ) :
	_unique_id   		   ( n_id ),
	_weight      		   ( 0 ),
	_scope_size  		   ( n_vars ),
	_args_size   		   ( n_args ),
	_vars        		   ( vars ),
	_args        		   ( args ),
	_additional_parameters ( nullptr ) {
	// Alloc memory for_status, _working_status, and indexes lookup array
	_status            = (uint**) malloc ( _scope_size * sizeof (uint*) );
  	_working_status    = (uint**) malloc ( _scope_size * sizeof (uint*) );
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
  free ( _working_status );
  free ( _status_idx_lookup );
}//~Constraint

__device__ void
CudaConstraint::move_status_to_shared (  uint * shared_ptr, int domain_size, int thread_offset )
{
    if ( shared_ptr == nullptr ) return;
	
    if ( blockDim.x * blockDim.y == 1 )
    {// One thread per block
        
        if ( threadIdx.x == 0 )
        {// Copy the (global) state into shared_ptr
        
            int idx = 0;
            for ( int i = 0; i < _scope_size; i++ )
            {
            	_working_status [ i ] = &shared_ptr [ idx ];
            	
            	// Failed event
                if ( _status [ i ][ EVT ] == FAL_EVT )
                {
                	_working_status [ i ][ EVT ] = FAL_EVT;
                	continue;
                }

                // Copy global memory into shared memory: read-only
                for ( int j = 0; j < domain_size; j++ )
                {
                    shared_ptr [ idx++ ] = _status [ i ][ j ];
            	}
            }//i 
        }
    }
    else 
    {// One thread per variable
    
    	int tid, loop   = 0;
        int num_threads = WARP_SIZE; //blockDim.x * blockDim.y;
        while ( (num_threads * loop) < _scope_size )
        {
            tid = thread_offset + threadIdx.x + num_threads * loop;
            
            ++loop;
            if ( tid < _scope_size )
            {
            	_working_status [ tid ] = &shared_ptr [ tid * domain_size ];
            	for ( int i = 0; i < domain_size; i++ )
            	{
            		shared_ptr [ tid * domain_size + i ] = _status [ tid ][ i ]; 
           		}
            }
        }
    }
}//move_status_to_shared

/*
 * ======================================================================
 * @note This function is deprecated and will be removed in the future.
 *       The user should avoid its use.
 * ======================================================================
 */
__device__ void
CudaConstraint::move_status_from_shared ( uint * shared_ptr, int domain_size, int ref, int thread_offset )
{
    if ( shared_ptr == nullptr ) return;

    if ( blockDim.x * blockDim.y == 1 )
    {// One thread per block
        
        if ( threadIdx.x == 0 )
        {
            int idx = 0;
            for ( int i = 0; i < _scope_size; i++ )
            {
                if ( ref >= 0 && i != ref ) 
                {
                	idx += domain_size;
                	continue;
                }
                
                // Check event
                if ( shared_ptr [ idx ] == FAL_EVT )
                {
                	_status [ i ][ EVT ] = FAL_EVT;
                    break;
                }
                else if ( shared_ptr [ idx ] == SNG_EVT || 
                		  shared_ptr [ idx ] == BOL_SNG_EVT )
                {
                    /*
                     * int old = atomicCAS ( &_temp_status [ i ][ EVT ], NOP_EVT, SNG_EVT );
                     * is translated into
                     *		(old == NOP_EVT ? SNG_EVT : old)
                     */
                    if ( domain_size != BOOLEAN_DOM )
                    {
                    	atomicCAS ( &_status [ i ][ EVT ], NOP_EVT, SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], BND_EVT, SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], MIN_EVT, SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], MAX_EVT, SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], CHG_EVT, SNG_EVT );
                    }
                    else 
                    {
                    	atomicCAS ( &_status [ i ][ EVT ], NOP_EVT, BOL_SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], BND_EVT, BOL_SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], MIN_EVT, BOL_SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], MAX_EVT, BOL_SNG_EVT );
                    	atomicCAS ( &_status [ i ][ EVT ], CHG_EVT, BOL_SNG_EVT );
                    }
                }
                else if ( domain_size == BOOLEAN_DOM )
                {// Boolean domain representation
                
                	atomicMin ( &_status [ i ][ ST ], shared_ptr [ idx + ST ] );
                    idx += BOOLEAN_DOM;
                    continue;
                }
                else
                {// Standard domain representation
					atomicCAS ( &_status [ i ][ EVT ], NOP_EVT, (uint)shared_ptr [ idx ] );  
                }
                
                // REP
                _status [ i ][ REP ] = shared_ptr [ idx + REP ];

                // LB
                atomicMax ( &_status [ i ][ LB ], shared_ptr [ idx + LB ] );

                // UB
                atomicMin ( &_status [ i ][ UB ], shared_ptr [ idx + UB ] );

                // DSZ
                atomicMin ( &_status [ i ][ DSZ ], shared_ptr [ idx + DSZ ] );

                // BIT
                idx += BIT;
                int dom_empty = 0;
                for ( int j = BIT; j < domain_size; j++ )
                {
                    dom_empty += ( (atomicAnd ( &_status [ i ][ j ], shared_ptr [ idx++ ] )) & shared_ptr [ idx-1 ] );
                }
				
				// Sanity check for race conditions
                if ( dom_empty == 0 )
                {
                    _status [ i ][ EVT ] = FAL_EVT;
                }
            }//i
        }
    }
    else if ( blockDim.x * blockDim.y >= _scope_size )
    {// One thread per variable
        
        int tid, loop   = 0;
        int num_threads = WARP_SIZE; //blockDim.x * blockDim.y;
        while ( (num_threads * loop) < _scope_size )
        {
            tid = thread_offset + threadIdx.x + num_threads * loop;
            loop++;
            
            if ( tid < _scope_size )
            {
                if ( ref >= 0 && tid != ref ) 
                {
                	break;
                }
                
                if ( shared_ptr [ tid * domain_size ] == FAL_EVT )
                {
                    _status [ tid ][ EVT ] = FAL_EVT;
                }
                else if ( shared_ptr [ tid * domain_size ] == SNG_EVT || 
                		  shared_ptr [ tid * domain_size ] == BOL_SNG_EVT )
                {
                	// (old == NOP_EVT ? SNG_EVT : old)
                	if ( domain_size != BOOLEAN_DOM )
                	{
                    	int old = atomicCAS ( &_status [ tid ][ EVT ], NOP_EVT, SNG_EVT );
                    	if ( old != FAL_EVT )
                    	{
                    		_status [ tid ][ EVT ] = SNG_EVT;
                    	}
                    }
                    else
                    {
                    	int old = atomicCAS ( &_status [ tid ][ EVT ], NOP_EVT, BOL_SNG_EVT );
                    	if ( old != FAL_EVT )
                    	{
                    		_status [ tid ][ EVT ] = BOL_SNG_EVT;
                    	}
                    }
                }
                else if ( domain_size != BOOLEAN_DOM )
                {// Standard domain representation
                    int tid_idx = tid * domain_size;
                    atomicMin ( &_status [ tid ][ EVT ], shared_ptr [ tid_idx ] );
                        
                    // REP
                    _status [ tid ][ REP ] = shared_ptr [ tid_idx + REP ];
                        
                    // LB
                    atomicMax ( &_status [ tid ][ LB ], shared_ptr [ tid_idx + LB ] );
                        
                    // UB
                    atomicMin ( &_status [ tid ][ UB ], shared_ptr [ tid_idx + UB ] );
                        
                    // DSZ
                    atomicMin ( &_status [ tid ][ DSZ ], shared_ptr [ tid_idx + DSZ ] );
                        
                    // BIT
                    tid_idx += BIT;
                    for ( int j = BIT; j < domain_size; j++ )
                    {
                        atomicAnd ( &_status [ tid ][ j ], shared_ptr [ tid_idx++ ] );
                    }
                }
                else
                {// Bool domain representation
                    atomicMin ( &_status [ tid ][ ST ], shared_ptr [ tid * domain_size + ST ] );
                }
            }
        }
    }
}//move_status_from_shared

__device__ void
CudaConstraint::move_bit_status_from_shared ( uint * shared_ptr, int d_size, int ref, uint* extern_status, int thread_offset )
{
    if ( shared_ptr == nullptr ) return;
    
    if ( blockDim.x * blockDim.y == 1 )
    {// One thread per block
    
        if ( threadIdx.x == 0 )
        {// Copy _working_status into _status or extern_status
        
        	/*
        	 * If extern_status is not NULL, 
        	 * copy _working_status into extern_status, 
        	 * otherwise copy it into _status.
        	 */
        	bool to_extern = ( extern_status != nullptr );
        	
            int idx = 0, ext_idx;
            for ( int i = 0; i < _scope_size; i++ )
            {
            	// Skip scope variables if ref is set
            	ext_idx = _status_idx_lookup [ i ];
                if ( ref >= 0 && i != ref ) 
                {
                	idx += d_size;
                	continue;
                }
                
                // Check event
                if ( shared_ptr [ idx ] == FAL_EVT )
                {
                	if ( to_extern )
                	{ 
                		extern_status [  ext_idx + EVT ] = FAL_EVT;
                    }
                    else
                    {
                    	_status [ i ][ EVT ] = FAL_EVT;
                    }
                }
                else if ( d_size == BOOLEAN_DOM )
                {// Bool representation
                
                	if ( to_extern  )
                	{
                		atomicMin ( &extern_status [ ext_idx +  ST ], shared_ptr [ idx + ST ] );
                    }
                    else
                    {
                    	atomicMin ( &_status [ i ][ ST ], shared_ptr [ idx + ST ] );
                    }
                    idx += BOOLEAN_DOM;
                    continue;
                }
                
                idx += BIT;
                for ( int j = BIT; j < d_size; j++ )
                {
                	if ( to_extern )
                	{
                		atomicAnd ( &extern_status [ ext_idx + j ], shared_ptr [ idx++ ] );
                    }
                    else
                    {
                    	atomicAnd ( &_status [ i ][ j ], shared_ptr [ idx++ ] );
                    }
                }
            }
        }
    }
    else
    {// One thread per variable

        int tid, loop = 0;
        bool to_extern  = ( extern_status != nullptr );
        int num_threads = WARP_SIZE; //blockDim.x * blockDim.y;
        while ( (num_threads * loop) < _scope_size )
        {
            tid = thread_offset + threadIdx.x + num_threads * loop;
            ++loop;
            
            if ( tid < _scope_size )
            {
            	// Skip if ref is set and the current block does not correspond to ref
                if ( ref >= 0 && tid != ref ) 
                {	
                	break;
                }
                
                
                // Check event
                if ( shared_ptr [ tid * d_size ] == FAL_EVT )
                {
                    if ( to_extern )
                	{
                		extern_status [ _status_idx_lookup [ tid ] + EVT ] = FAL_EVT;
                    }
                	else
                    {
                        _status [ tid ][ EVT ] = FAL_EVT;
                    }
                }
                else if ( d_size == BOOLEAN_DOM )
                {//Bool domain representation
                
                    if ( to_extern )
                	{
                		atomicMin ( &extern_status [ _status_idx_lookup [ tid ] + ST ], shared_ptr [ tid * d_size + ST ] );
                    }
                    else
                    {
                        atomicMin ( &_status [ tid ][ ST ], shared_ptr [ tid * d_size + ST ] );
                    }
                }
                else
                {// Standard domain representation 
                
                    int tid_idx = tid * d_size;
                    tid_idx += BIT;
                    for ( int j = BIT; j < d_size; j++ )
                    {
                        if ( to_extern )
                		{
                			atomicAnd ( &extern_status [ _status_idx_lookup [ tid ] + j ], shared_ptr [ tid_idx++ ] );
                        	
                        }
                        else
                        {
                        	atomicAnd ( &_status [ tid ][ j ], shared_ptr [ tid_idx++ ] );
                        }
                    }
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
        if ( _working_status[ i ][ EVT ] != SNG_EVT &&
        	 _working_status[ i ][ EVT ] != BOL_SNG_EVT)
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
        if ( _working_status[ i ][ EVT ] != SNG_EVT && 
        	 _working_status[ i ][ EVT ] != BOL_SNG_EVT )
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
    return (_working_status[ var ][ EVT ] == SNG_EVT || _working_status[ var ][ EVT ] == BOL_SNG_EVT );
}//is_singleton

__device__ bool
CudaConstraint::is_ground ( int var ) const
{
    return (_working_status[ var ][ EVT ] == SNG_EVT || _working_status[ var ][ EVT ] == BOL_SNG_EVT );
}//is_ground

__device__ int
CudaConstraint::get_not_ground () const
{
	for ( int i = 0; i < _scope_size; i++ )
        if ( _working_status[ i ][ EVT ] != SNG_EVT && 
        	 _working_status[ i ][ EVT ] != BOL_SNG_EVT )
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
    if ( _working_status[ var ][ EVT ] == FAL_EVT ) return; 
    
    // Boolean domain
    if ( _working_status[ var ][ EVT ] == BOL_EVT || 
    	 _working_status[ var ][ EVT ] == BOL_SNG_EVT )
    {
        if ( val < 0 || val > 1 ) return;
        
        int b_val = _working_status[ var ][ ST ];
        if ( b_val == val )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _working_status[ var ][ EVT ] = FAL_EVT;
        }
        else
        {// b_val is BOL_U (undef) -> remove one val and set singleton
            _working_status[ var ][ ST ]   = val;
            _working_status[ var ][ EVT ]  = BOL_SNG_EVT;
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _working_status [ var ][ LB ];
        int upper_bound = _working_status [ var ][ UB ];
        
        if ( val < lower_bound || val > upper_bound ) return;
        if ( _working_status [ var ][ REP ] == 0 )
        {// Bitmap representation for domain
            /*
             * Find the chunk and the position of the bit within the chunk.
             * @note: chunks are stored in reversed/LSB order.
             *   For example: {0, 63} is stored as
             *        | 0 | 0 | 0 | 0 | 0 | 0 | 63...32 | 31...0 |
             */
            int chunk = val / BITS_IN_CHUNK;
            chunk = BIT + NUM_CHUNKS - 1 - chunk;
            uint val_chunk = _working_status[ var ][ chunk ];
            uint val_clear = val % BITS_IN_CHUNK;
            
            // Check if the element val is in the domain, if not return
            if ( !((val_chunk & ( 1 << val_clear )) != 0) ) return;
            
            // If val belongs to the domain, subtract it
            _working_status[ var ][ chunk ] = val_chunk & (~(1 << val_clear ));
            
            // Make the size consistent (avoid race conditions)
            int domain_size = 0;
            for ( int i = BIT; i < BIT + NUM_CHUNKS; i++ )
            {
            	if ( _status[ var ][ i ] == 0 ) continue;
            	domain_size += num_1bit ( _working_status[ var ][ i ] );
            }
            
            // Empty domain: fail event
            if ( domain_size <= 0 )
            {// Failed event
                _working_status[ var ][ EVT ] = FAL_EVT;
                return;
            }
			
			// Singleton event
            if ( domain_size == 1 )
            {
            	_working_status[ var ][ DSZ ] = 1;
            	
                // Lower bound increased: val was the lower bound
                if ( lower_bound == val )
                {
                    _working_status[ var ][ LB ]  = upper_bound;
                    _working_status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                // Upper bound decreased: val was the upper bound
                _working_status[ var ][ UB ]  = lower_bound;
                _working_status[ var ][ EVT ] = SNG_EVT;
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
                        _working_status[ var ][ LB ]  = lower_bound;
                        break;
                    }
                }
                _working_status[ var ][ EVT ] = MIN_EVT;
            }
            else if ( upper_bound == val )
            {// Upper bound decreased
            
            	while ( upper_bound >= lower_bound ) 
            	{// Find new upper bound
                    --upper_bound;
                    if ( contains ( var, upper_bound ) ) 
                    {
                        _working_status[ var ][ UB ]  = upper_bound;
                        break;
                    }
                }
            	_working_status[ var ][ EVT ] = MAX_EVT;
            }
            else
            {
            	_working_status[ var ][ EVT ] = CHG_EVT;
            }
            
            // Set new domain size
            _working_status[ var ][ DSZ ] = domain_size;
        }
        else
        {// Pair of bounds
            if ( val > lower_bound && val < upper_bound ) return;
            if ( lower_bound == val )
            {
                if ( upper_bound == val )
                {
                    _working_status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( upper_bound == val+1 )
                {
                    _working_status[ var ][ LB ]  = upper_bound;
                    _working_status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                _working_status[ var ][ LB ]  = lower_bound + 1;
                _working_status[ var ][ EVT ] = BND_EVT;
                return;
            }
            else if ( upper_bound == val )
            {
                if ( lower_bound == val )
                {
                    _working_status[ var ][ EVT ] = FAL_EVT;
                    return;
                }
                if ( lower_bound == val-1 )
                {
                    _working_status[ var ][ UB ]  = lower_bound;
                    _working_status[ var ][ EVT ] = SNG_EVT;
                    return;
                }
                
                _working_status[ var ][ UB ]  = upper_bound - 1;
                _working_status[ var ][ EVT ] = BND_EVT;
                return;
            }
        }
    }
}//subtract

__device__ int
CudaConstraint::get_min ( int var ) const
{
    if ( _working_status[ var ][ EVT ] == BOL_EVT ||
    	 _working_status[ var ][ EVT ] == BOL_SNG_EVT )
    {
        int val = _working_status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_F;
        }
        return val;
    }

    // Standard domain representation
    return _working_status[ var ][ LB ];
}//get_min

__device__ int
CudaConstraint::get_max ( int var ) const
{
    if ( _working_status[ var ][ EVT ] == BOL_EVT || 
    	 _working_status[ var ][ EVT ] == BOL_SNG_EVT )
    {
        int val = _working_status[ var ][ ST ];
        if ( val == BOL_U )
        {
            return BOL_T;
        }
        return val;
    }

    // Standard domain representation
    return _working_status[ var ][ UB ];
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
    return ( _working_status[ var ][ EVT ] == FAL_EVT );
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
    if ( _working_status[ var ][ EVT ] == BOL_EVT || _working_status[ var ][ EVT ] == BOL_SNG_EVT )
    {
        if ( smin < 0 || smin > 1 || smax < 0 || smax > 1 || smax < smin ) return;
        if ( smin + smax == 1 )
        {// b_val is 0 or 1 and status is 0 or 1 -> fail
            _working_status[ var ][ EVT ] = FAL_EVT;
        }
        if ( smin == smax )
        {
            subtract ( var, smin );
        }
    }
    else
    {
        // @note Only one pair of bounds is allowed
        int lower_bound = _working_status [ var ][ LB ];
        int upper_bound = _working_status [ var ][ UB ];
        if ( smin <= lower_bound && smax >= upper_bound ) return;
        if ( smin > smax )
        {
            _working_status[ var ][ EVT ] = FAL_EVT;
            return;
        }
        
        smin = (smin > lower_bound) ? smin : lower_bound;
        smax = (smax < upper_bound) ? smax : upper_bound;
        
        if ( smin == smax )
        {
            _working_status[ var ][ DSZ ] = 1;
            _working_status[ var ][ LB ]  = smin;
            _working_status[ var ][ UB ]  = smin;
            _working_status[ var ][ EVT ] = SNG_EVT;
            return;
        }
        
        if ( _working_status [ var ][ REP ] == BIT_REP )
        {// Bitmap representation for domain
        
            int chunk_min = smin / BITS_IN_CHUNK;
            chunk_min = BIT + NUM_CHUNKS - 1 - chunk_min;
            int chunk_max = smax / BITS_IN_CHUNK;
            chunk_max = BIT + NUM_CHUNKS - 1 - chunk_max;
            for ( int i = BIT; i < chunk_min; i++ )                  _working_status [ var ][ i ] = 0;
            for ( int i = BIT + NUM_CHUNKS - 1; i > chunk_max; i-- ) _working_status [ var ][ i ] = 0;
            clear_bits_i_through_0   ( _working_status [ var ][ chunk_min ], (smin % BITS_IN_CHUNK) - 1 );
            clear_bits_MSB_through_i ( _working_status [ var ][ chunk_max ], (smax % BITS_IN_CHUNK) + 1 );

            int num_bits = 0;
            for ( int i = chunk_min; i <= chunk_max; i++ )
                num_bits += num_1bit ( (uint) _working_status [ var ][ i ] );

            if ( num_bits == 0 )
            {
                _working_status[ var ][ EVT ] = FAL_EVT;
                return;
            }
            if ( num_bits == 1 )
            {
                _working_status[ var ][ DSZ ] = 1;
                _working_status[ var ][ LB ]  = smin;
                _working_status[ var ][ UB ]  = smin;
                _working_status[ var ][ EVT ] = SNG_EVT;
                return;
            }
            
            _working_status[ var ][ DSZ ] = num_bits;
            lower_bound = (smin < lower_bound) ? lower_bound : smin;
            upper_bound = (smax > upper_bound) ? upper_bound : smax;
            
            while ( !contains ( var, lower_bound ) ) lower_bound++;
    		while ( !contains ( var, upper_bound ) ) upper_bound--;
            	
            _working_status[ var ][ LB ]  = lower_bound;
            _working_status[ var ][ UB ]  = upper_bound;
            _working_status[ var ][ EVT ] = CHG_EVT;
        }
        else
        {// Pair of bounds
        
            if ( smin > lower_bound && smax > upper_bound )
            {
                int cnt = smin - lower_bound;
                _working_status[ var ][ LB ] = smin;
                _working_status[ var ][ DSZ ] -= cnt;
                _working_status[ var ][ EVT ] = BND_EVT;
                return;
            }
            if ( smin < lower_bound && smax < upper_bound )
            {
                int cnt = upper_bound - smax;
                _working_status[ var ][ UB ] = smax;
                _working_status[ var ][ DSZ ] -= cnt;
                _working_status[ var ][ EVT ] = BND_EVT;
                return;
            }
            int cnt = (smin - lower_bound) + (upper_bound - smax);
            _working_status[ var ][ LB ] = smin;
            _working_status[ var ][ UB ] = smax;
            _working_status[ var ][ DSZ ] -= cnt;
            _working_status[ var ][ EVT ] = BND_EVT;
        }
    }
}//shrink

__device__ bool 
CudaConstraint::contains ( int var, int val )
{
	if ( _working_status[ var ][ EVT ] == BOL_EVT || 
	 	 _working_status[ var ][ EVT ] == BOL_SNG_EVT )
    {
        int val_in = _working_status[ var ][ ST ];
        if ( val_in == BOL_U )
        {
            return (val >= 0 && val <= 1);
        }
        return val == val_in;
    }
    
    int chunk = val / BITS_IN_CHUNK;
    chunk = BIT + NUM_CHUNKS - 1 - chunk;
	return ( (_working_status [ var ][ chunk ] & (1 << (val % BITS_IN_CHUNK))) != 0 );
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

__device__ bool 
CudaConstraint::is_soft () const
{
	return _weight > 0;
}//is_soft

__device__ int 
CudaConstraint::unsat_level () const
{
	printf ( "CudaConstraint::unsat_level - not yet implemented\n" );
	return 0;
}//unsat_level

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

__device__ void 
CudaConstraint::set_additional_parameters ( void* additional_parameters_ptr )
{	
	_additional_parameters = static_cast < int* >( additional_parameters_ptr );
}//set_additional_parameters

#endif
