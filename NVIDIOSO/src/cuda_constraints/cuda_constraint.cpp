//
//  cuda_constraint.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 02/12/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//

#include "cuda_constraint.h"

#if CUDAON
__device__
CudaConstraint::CudaConstraint ( int n_id, int n_vars, int n_args,
                                 int* vars, int* args,
                                 int* domain_idx, uint* domain_states ) :
_unique_id  ( n_id ),
_scope_size ( n_vars ),
_args_size  ( n_args ),
_vars       ( vars ),
_args       ( args ) {
  _status = (uint**) malloc ( n_vars * sizeof (uint*));
  for ( int i = 0; i < n_vars; i++ )
  {
    _status[ i ] = &domain_states[ domain_idx[ _vars[ i ] ] ];
  }//i
}//CudaConstraint

__device__
CudaConstraint::~CudaConstraint () {
  free ( _vars );
  free ( _args );
  free ( _status );
}//~Constraint


__device__ bool 
CudaConstraint::all_ground () const
{
    for ( int i = 0; i < _scope_size; i++ )
        if ( _status[ i ][ EVT ] != SNG_EVT )
            return false;
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
CudaConstraint::subtract ( int var, int val )
{
    /*
     * Three representations of domains to consider:
     * 1) Bitmap for non Boolean vars
     * 2) Bounds for non Boolean vars
     * 3) Bits for Boolean vars
     */
}//subtract

__device__ int
CudaConstraint::get_min ( int var ) const
{
    return _status[ var ][ LB ];
}//get_min

__device__ int
CudaConstraint::get_max ( int var ) const
{
    return _status[ var ][ UB ];
}//get_max

__device__ bool
CudaConstraint::is_empty ( int var ) const
{
    return (_status[ var ][ EVT ] == FAL_EVT);
}//is_empty

__device__ void
CudaConstraint::shrink ( int var, int min, int max )
{
    /*
     * Three representations of domains to consider:
     * 1) Bitmap for non Boolean vars
     * 2) Bounds for non Boolean vars
     * 3) Bits for Boolean vars
     */
}//shrink

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
