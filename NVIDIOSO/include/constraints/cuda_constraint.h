//
//  cuda_constraint.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 02/12/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface/abstract class for all constraints that run on CUDA device.
//  Defines how to construct a constraint, impose, check satisiability,
//  enforce consistency, etc.
//

#ifndef NVIDIOSO_cuda_constraint_h
#define NVIDIOSO_cuda_constraint_h

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cstddef>

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_constraint_utility.h"
#endif

using uint = unsigned int;

class CudaConstraint {
protected:
  //! Unique global identifier for a given constraint.
  size_t _unique_id;
  
  //! Scope size
  int _scope_size;
  
  /**
   * It represents the array of pointers to
   * the domains of the variables in
   * the scope of this constraint.
   */
  int* _vars;
  
  //! Number of arguments
  int _args_size;
  
  /**
   * It represents the array of auxiliary arguments needed by
   * a given constraint in order to be propagated.
   * For example:
   *    int_eq ( x, 2 ) has 2 as auxiliary argument.
   */
  int* _args;
  
  /**
   * Array of pointers to the domains 
   * of the variables involved in this constraint.
   */
  uint** _status;

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
  __device__ CudaConstraint ( int n_id, int n_vars, int n_args,
                              int* vars, int* args,
                              int* domain_idx, uint* domain_states );
#endif
  
public:
  
#if CUDAON
  
  __device__ virtual ~CudaConstraint ();
  
  //! Get unique (global) id of this constraint.
  __device__ size_t get_unique_id () const;
  
  /**
   * Get the size of the scope of this constraint,
   * i.e., the number of FD variables which is defined on.
   * @note The size of the scope does not correspond to the formal
   *       definition of the constraint but with the actual number
   *       of variables within the scope of a given constraint.
   *       For example:
   *          int_eq ( x, y ) has _scope_size equal to 2;
   *          int_eq ( x, 1 ) has _scope_size equal to 1.
   */
  __device__ size_t get_scope_size () const;
  
  //!Get the size of the auxiliary arguments of this constraint.
  __device__ size_t get_arguments_size () const;

  /**
   * It is a (most probably incomplete) consistency function which
   * removes the values from variable domains. Only values which
   * do not have any support in a solution space are removed.
   */
  __device__ virtual void consistency () = 0;
  
  /**
   * It checks if the constraint is satisfied.
   * @return true if the constraint if for certain satisfied,
   *         false otherwise.
   * @note If this function is incorrectly implementd, 
   *       a constraint may not be satisfied in a solution.
   */
  __device__ virtual bool satisfied () = 0;
  
  
  //! Prints info.
  __device__ virtual void print () const = 0;
#endif
  
};

#endif
