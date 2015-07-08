//
//  cuda_set_ne.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/07/15.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  Constraint
//  Domain consistency is used.
//

#ifndef __NVIDIOSO__cuda_set_ne__
#define __NVIDIOSO__cuda_set_ne__

#include "cuda_constraint.h"

class CudaSetNe : public CudaConstraint {
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
  __device__ CudaSetNe ( int n_id, int n_vars, int n_args,
                         int* vars, int* args,
                         int* domain_idx, uint* domain_states );
  
  __device__ ~CudaSetNe ();
  
  //! It performs domain consistency
  __device__ void consistency ();
  
  //! It checks if
  __device__ bool satisfied ();
  
  //! Prints the semantic of this constraint
  __device__ void print () const;
#endif
  
};

#endif
