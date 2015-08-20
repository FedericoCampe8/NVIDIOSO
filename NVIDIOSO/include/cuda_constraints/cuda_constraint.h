//
//  cuda_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 12/02/14.
//  Modified by Federico Campeotto in 08/01/15
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class represents the interface/abstract class for constraints on device.
//  Defines how to declare and define a constraint, check for satisiability,
//  enforce consistency, etc.
//  Domain representation for standard domains and Boolean domains:
//  - | EVT | REP | LB | UB | DSZ | Bit
//  - | EVT | BOOL|
//  where BOOL can be 0 (False), 1 (True), 2 (Undef-Not ground).
//  @note In the current implementation we consider only the above two representations
//        which are mutually exclusive.
//  @note Constraints work directly on arrays representing the domain of a variable.
//        The use of arrays gives to the user the ability to implement different 
//        search strategies (e.g., parallel beam search) by passing directly the arrays 
//        representing the current status of the computation.
//  @note Each constraint has an array of pointer (_status) pointing to the array in 
//        global memory containing all the status (i.e., domains) of all the variables
//        in the model.
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
  
  	/**
  	 * Weight of this constraint.
  	 * This may be used, for example, for soft constraints.
  	 * @note default is 0 and it identifies a hard constraint.
  	 *       A value grater than 0 identifies a soft constraint.
  	 */
  	 int _weight;
  
    //! Scope size
    int _scope_size;
  
    /**
     * Array of pointers to
     * the indexes of the variables in
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
     * Array of additional parameters needed by the constraint
     * in order to be propagated (e.g., table constraint).
     * @note All additional parameters are serialized into this
     *       array and each subarray begins with two values:
     *       1 - Number of rows
     *       2 - Number of columns
     *       Therefore, 
     *       an array of length n is | 1 | n | a_1 | a_2 | ... | a_n |,
     *       an n*m matrix is | n | m | a_11 | a_12 | ... | a_nm |.
     */
  	 int* _additional_parameters;	
  		
    /**
     * Mapping between indexes of vars in this constraint
     * as 0, 1, 2, ... and the corresponding indexes on the global array
     * of the status of all the variables.
     */
     int * _status_idx_lookup;
     
    /**
     * Array of pointers to the domains 
     * of the variables involved in this constraint.
     * Constraint propagation can be performed either on this array
     * or (better) on shared memory and use this array as a read-only array.
     */
    uint** _status;

    /**
     * Temporary status (pointers to) used for copies to/from shared memory.
     * This array can be used as "temporary output" array, i.e., result of 
     * domain propagation can be set on this write-only array.
     * @note This is a temporary array, after domain propagation, results must be
     *       consistent on the array _status.
     * @note Propagation algorithms WORK ON THIS array. 
     *       Therefore, _status array must be copied (pointers to) to _working_status
     *       before ANY propagation happen.
     * @note By default this array points to NULL.
     */
    uint** _working_status;
    
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
                                
    /*=======================================================
     *
     *				DOMAIN INFO FUNCTIONS
     *
     * The following functions get information  about
     * the current status of the variables 
     * in the constraint's scope. 
     * These functions DO NOT use atomic operations, i.e., 
     * if the domain is shared between concurrent threads, 
     * the program may suffer of race conditions.
     * Please, use shared memory for local changes w.r.t. 
     * different block or synchronization utilities w.r.t. 
     * threads in the same block.
     * @note The following functions work on the
     *       		_working_status 
     *       array.
     *======================================================*/
    
    //! Returns True if all variables are singletons, False otherwise
    __device__ bool all_ground () const;
    
    //! Returns True if all variables but one are singletons, False otherwise
    __device__ bool only_one_not_ground () const;
    
    //! Returns True if the variable var is singleton, False otherwise
    __device__ bool is_singleton ( int var ) const;

    //! Returns True if the variable var is ground, False otherwise
    __device__ bool is_ground ( int var ) const;

    /**
     * Returns the first var (idx) which is not ground.
     * @note it returns -1 if all variables are ground.
     */
    __device__ int get_not_ground () const;
    
    /**
     * Returns True if val belongs to the domain of var, False otherwise.
     * @note This function DOES NOT perform any check on the given value val.
     */
    __device__ bool contains ( int var, int val );
	
    //! Get lower bound of the domain of var
    __device__ int get_min ( int var ) const;

    //! Get upper bound of the domain of var
    __device__ int get_max ( int var ) const;
    
    //! Get sum of ground vars * aux elements
    __device__ int get_sum_ground () const;
    
    //! Returns true if the domain of var is empty
    __device__ bool is_empty ( int var ) const;

    //! Returns the number of set bits in val
    __device__ int  num_1bit ( uint val );

    /*=======================================================
     *
     *				DOMAIN MANIPULATION FUNCTIONS
     *
     * The following functions manipulate the domain of 
     * the variables in the constraint's scope. 
     * These functions DO NOT use atomic operations, i.e., 
     * if the domain is shared between concurrent threads, 
     * the program may suffer of race conditions.
     * Please, use shared memory for local changes w.r.t. 
     * different block or synchronization utilities w.r.t. 
     * threads in the same block.
     * @note The following functions work on the
     *       		_working_status 
     *       array.
     *======================================================*/
     
    //! Clears all set bits from pos i to pos 0
    __device__ void clear_bits_i_through_0   ( uint& val, int idx );

    //! Clears all set bits from MSB to pos 0
    __device__ void clear_bits_MSB_through_i ( uint& val, int idx );
    
    //! Subtract {val} from the domain of var
    __device__ void subtract ( int var, int val, int ref = -1 );

    //! Shrinks the domain of var to {min, max}
    __device__ void shrink ( int var, int min, int max, int ref = -1 );
    
#endif
  
public:
  
#if CUDAON
  
    __device__ virtual ~CudaConstraint ();
  
    //! Get unique (global) id of this constraint.
    __device__ size_t get_unique_id () const;
    
    //! Is soft information
    __device__ bool is_soft () const;
  
  	/**
   	 * It returns an integer value that can be used
     * to represent how much the current constraint is
     * unsatisfied. This function can be used to
     * implement some heuristics for optimization problems.
     * @return an integer value representing how much this 
     *         constraint is unsatisfied. It returns 0 if
     *         this constraint is satisfied.
     */
  	__device__ int unsat_level () const;
  
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
     * @param thread_offset an offset may be needed to copy status in parallel on shared.
     * @note No synchronization or atomic operatations are used here.
     * @todo If shared_ptr is NULL use _status instead of shared_ptr.
     */
    __device__ void move_status_to_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM, int thread_offset = 0  );
    
    /**
     * === DEPRECATED ===
     * Copy status from shared memory to global memory.
     * @param shared_ptr pointer to the region of shared memory
     *        where status will be copied from.
     * @param size of domains: standard (n ints), Boolean (2 ints), mixed.
     * @param ref reference to the variable to move from shared (default: all variables in the scope).
     * @param thread_offset an offset may be needed to copy status in parallel on shared.
     * @note This function uses atomic operations to synchronize writes on gloabl memory
     */
    __device__ void move_status_from_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM, int ref = -1, int thread_offset = 0 );

    /**
     * Copy the domains from shared to global memory.
     * It copies ONLY the bitmap status (or the bounds) WITHOUT updating
     * EVT, LB, UP, DSZ, etc. fields.
     * @param shared_ptr pointer to the region of shared memory
     *        where status will be copied from.
     * @param size of domains: standard (n ints), Boolean (2 ints), mixed.
     * @param ref reference to the variable to move from shared (default: all variables in the scope)
     * @param extern_status where to copy back the values from shared memory. 
     *        If nullptr use pre-assigned global memory.
     * @param thread_offset an offset may be needed to copy status in parallel on shared.
     * @note Updates on the bitmap field is performed with atomic operations.
     * @note This functions uses less atomics than move_status_from_shared and therefore
     *       if faster on moving from shared to global.
     *       However, non-bit fields must be updated accordingly.
     */
    __device__ void move_bit_status_from_shared ( uint * shared_ptr = nullptr, int d_size = MIXED_DOM, int ref = -1, uint* extern_status = nullptr, int thread_offset = 0 );
    
    /**
	 * Set additional parameters array.
	 * @param additional_parameters_ptr (void) pointer to the area on device
	 *        storing all additional parameters needed by the constraint.
	 * @note any additional parameter array needed by this constraint
	 *       can be set using this method as long as it has the format
	 *       | # rows | # columns | values |.
	 * @note by default additional parameters array is a pointer pointing to NULL and
	 *       additional_parameters_ptr is converted to a pointer to integer.
	 */
    __device__ virtual void set_additional_parameters ( void* additional_parameters_ptr );
    
    /**
     * It is a (most probably incomplete) consistency function which
     * removes the values from variable domains. Only values which
     * do not have any support in a solution space are removed.
     * @param ref reference (index 0, 1, 2, ... ) of the variable in
     *        the scope of this constraint on which consistency will be performed.
     * @note by default ref if -1 meaning that consistency will be performed
     *       (possibly) on all the variables in the scope of this constraint.
     */
    __device__ virtual void consistency ( int ref = -1 ) = 0;
    
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
