//
//  cuda_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 12/02/14.
//  Modified by Federico Campeotto in 07/07/15
//  Copyright (c) 2014-2015 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface/abstract class for all constraints that run on CUDA device.
//  Defines how to construct a constraint, impose, check satisiability,
//  enforce consistency, etc.
//  Domain representation for standard domains and Boolean domains:
//  - EVT | REP | LB | UB | DSZ | Bit
//  - EVT | BOOL|
//  where BOOL can be 0 (False), 1 (True), 2 (Undef-Not ground)
//

#ifndef NVIDIOSO_cuda_constraint_h
#define NVIDIOSO_cuda_constraint_h

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cstddef>

#include "cuda_constraint_macro.h"

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

    //! Temporary status used for copy on shared memory
    uint** _temp_status;
    
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
	
    //! Returns True if val belongs to the domain of var, False otherwise
    __device__ bool contains ( int var, int val );
	
    //! Get lower bound of the domain of var
    __device__ int get_min ( int var ) const;

    //! Get upper bound of the domain of var
    __device__ int get_max ( int var ) const;
    
    //! Get sum of ground vars * aux elements
    __device__ int get_sum_ground () const;
    
    //! Returns true if the domain of var is empty
    __device__ bool is_empty ( int var ) const;

    //! Shrinks the domain of var to {min, max}
    __device__ void shrink ( int var, int min, int max );

    //! Utility for bit manipulation
    __device__ void clear_bits_i_through_0   ( uint& val, int idx );

    //! Utility for bit manipulation
    __device__ void clear_bits_MSB_through_i ( uint& val, int idx );

    //! Utility for bit manipulation
    __device__ int  num_1bit ( uint val );
    
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
     * Copy status (domains) from global to shared memory.
     * @param shared_ptr pointer to the region of shared memory
     *        where status in global memory will be copied to.
     * @param size of domains: standard (n ints), Boolean (2 ints), mixed.
     * @note No synchronization or atomic operatations are used here
     */
    __device__ void move_status_to_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM );
    
    /**
     * Copy status from shared memory to global memory.
     * @param shared_ptr pointer to the region of shared memory
     *        where status will be copied from.
     * @param size of domains: standard (n ints), Boolean (2 ints), mixed.
     * @note This function uses atomic operations to synchronize writes on gloabl memory
     */
    __device__ void move_status_from_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM );

    /**
     * Copy the domains from shared to global memory.
     * It copies only the bitmap status (or the bounds) without
     * updating the EVT, LB, UP, DSZ, etc. fields.
     * @param shared_ptr pointer to the region of shared memory
     *        where status will be copied from.
     * @param size of domains: standard (n ints), Boolean (2 ints), mixed.
     * @note Updates on the bitmap field is performed with atomic operations.
     * @note This functions uses less atomics than move_status_from_shared and therefore
     *       if faster on moving from shared to global.
     *       However, the non-bit fields should be updated somewhere else.
     */
    __device__ void move_bit_status_from_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM );
    
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
