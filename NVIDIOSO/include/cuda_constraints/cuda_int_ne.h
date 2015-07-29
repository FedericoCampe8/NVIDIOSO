//
//  cuda_int_ne.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Constraint X #\= Y.
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__cuda_int_ne__
#define __NVIDIOSO__cuda_int_ne__

#include "cuda_constraint.h"

class CudaIntNe : public CudaConstraint {
public:

#if CUDAON
  /**
   * Constructor.
   * @param n_id id of this constraint
   * @param n_vars scope size
   * @param n_args number of auxiliary arguments
   * @param vars pointer to the area of device memory storing
   *        the list of id variables involved in this constraint
   * @param args pointer to the area of memory containing auxiliary values
   * @param domain_idx indeces of the starting point of each variable domain
   *        in the array domain_states
   * @param domain_states array containing all variable domains
   */
  __device__ CudaIntNe ( int n_id, int n_vars, int n_args,
                         int* vars, int* args,
                         int* domain_idx, uint* domain_states );
  
  __device__ ~CudaIntNe ();
  
  //! It performs domain consistency
  __device__ void consistency ( int ref = -1 );
  
  //! It checks if x != y
  __device__ bool satisfied ();
  
  //! Prints the semantic of this constraint
  __device__ void print () const;
#endif
  
};

#endif /* defined(__NVIDIOSO__int_ne__) */
