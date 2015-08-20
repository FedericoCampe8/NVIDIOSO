//
//  cuda_global_constraint.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 19/08/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class is the interface/abstract class for global constraints on device.
//  Defines how to declare and define a global constraint, check for satisiability,
//  enforce consistency, etc.
//  This class derives from CudaConstraint and therefore it inherits its methods.
//

#ifndef NVIDIOSO_cuda_global_constraint_h
#define NVIDIOSO_cuda_global_constraint_h

#include "cuda_constraint.h"

class CudaGlobalConstraint : public CudaConstraint {
protected:

	//! Number of parallel blocks (i.e., groups of threads) 
	std::size_t _num_blocks;
  
  	//! Number of parallel threads per block
  	std::size_t _num_threads;
  	
  	/**
  	 * Propagation type:
  	 * 0 -> naive
  	 * 1 -> bound
  	 * 2 -> full
  	 * @note default is naive.
  	 */
  	 int _consistency_type;
    
#if CUDAON

	//! Naive consistency algorithm
  	__device__ virtual void naive_consistency () = 0;
  
  	//! Bound consistency algorithm
  	__device__ virtual void bound_consistency () = 0;
  
  	//! Full consistency algorithm
  	__device__ virtual void full_consistency  () = 0;
  
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
    __device__ CudaGlobalConstraint ( int n_id, int n_vars, int n_args,
                                	  int* vars, int* args,
                                	  int* domain_idx, uint* domain_states,
                                	  int num_blocks = 1, int num_threads = 1 );
                                
#endif
  
public:
  
#if CUDAON
  
    __device__ ~CudaGlobalConstraint ();
  
  	__device__ void set_consistency_type ( int con_type );
  	
    /**
     * It is a (most probably incomplete) consistency function which
     * removes the values from variable domains. Only values which
     * do not have any support in a solution space are removed.
     * @param ref reference (index 0, 1, 2, ... ) of the variable in
     *        the scope of this constraint on which consistency will be performed.
     * @note by default ref if -1 meaning that consistency will be performed
     *       (possibly) on all the variables in the scope of this constraint.
     */
    __device__ void consistency ( int ref = -1 ) override;
    
    /**
     * It checks if the constraint is satisfied.
     * @return true if the constraint if for certain satisfied,
     *         false otherwise.
     * @note If this function is incorrectly implementd, 
     *       a constraint may not be satisfied in a solution.
     */
    __device__ bool satisfied () override;
  
  
    //! Prints info.
    __device__ void print () const override;
    
#endif
  
};

#endif
