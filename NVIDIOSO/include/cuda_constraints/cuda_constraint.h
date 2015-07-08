//
//  cuda_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 02/12/14.
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
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

// EVENTS ON DOMAINS
#define NOP_EVT 0
#define SNG_EVT 1
#define BND_EVT 2
#define MIN_EVT 3
#define MAX_EVT 4
#define CHG_EVT 5
#define FAL_EVT 6

// INDEX ON DOMAINS
#define EVT 0
#define REP 1
#define LB  2
#define UB  3
#define DSZ 4

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
     * @param n_id unique id for the constraint
     * @param n_vars scope size, i.e., number of involved variables
     * @param n_args number of auxiliary arguments
     * @param vars pointer to the area of device memory storing
     *        the list of id variables involved in this constraint
     * @param args pointer to the area of memory containing the auxiliary arguments
     * @param domain_idx lookup array containing the indeces
     *        of the domains (stored in domains_states) of all the variables
     *        declared in the model.
     * @param domain_states array of all the domains stored sequentially in memory
     * @note domain_idx can be used to lookup the domains corresponding to the
     *       (ids of the) variables stored in vars
     */
    __device__ CudaConstraint ( int n_id, int n_vars, int n_args,
                                int* vars, int* args,
                                int* domain_idx, uint* domain_states );
    
    //! Returns True if all variables are singletons, False otherwise
    __device__ bool all_ground () const;
    
    //! Returns True if all variables but one are singletons, False otherwise
    __device__ bool only_one_not_ground () const;
    
    //! Returns True if the variable var is singleton, False otherwise
    __device__ bool is_singleton ( int var ) const;

    //! Returns True if the variable var is ground, False otherwise
    __device__ bool is_ground ( int var ) const;

    //! Returns the first var (idx) which is not ground
    __device__ int get_not_ground () const;

    //! Subtract {val} from the domain of var
    __device__ void subtract ( int var, int val );

    //! Get lower bound of the domain of var
    __device__ int get_min ( int var ) const;

    //! Get upper bound of the domain of var
    __device__ int get_max ( int var ) const;

    //! Returns true if the domain of var is empty
    __device__ bool is_empty ( int var ) const;

    //! Shrinks the domain of var to {min, max}
    __device__ void shrink ( int var, int min, int max );
    
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
