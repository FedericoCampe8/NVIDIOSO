//
//  cuda_alldifferent.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 19/08/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class specializes cuda global constraints for alldifferent.
//  There are three different propagation algorithms (i.e., three propagators)
//  that can be used:
//  1 - Eliminate singletons by naive value propagation.
//  2 - Bounds consistent alldifferent propagator.
//      Algorithm taken from:
//      	A. Lopez-Ortiz, C.-G. Quimper, J. Tromp, and P. van Beek.
//        	A fast and simple algorithm for bounds consistency of the
//        	alldifferent constraint. IJCAI-2003.
//
//      This implementation uses the code that is provided by Peter Van Beek:
//			http://ai.uwaterloo.ca/~vanbeek/software/software.html
//  3 - Domain consistent distinct propagator.
//		The algorithm is taken from:
// 			Jean-Charles RÈgin, A filtering algorithm for constraints
//		    of difference in CSPs, Proceedings of the Twelfth National
//			Conference on Artificial Intelligence, pages 362--367.
//			Seattle, WA, USA, 1994.
//


#ifndef __NVIDIOSO__cuda_alldifferent__
#define __NVIDIOSO__cuda_alldifferent__

#include "cuda_global_constraint.h"

class CudaAlldifferent : public CudaGlobalConstraint {
protected:

#if CUDAON

	//! Naive consistency algorithm
  	__device__ void naive_consistency () override;
  
  	//! Bound consistency algorithm
  	__device__ void bound_consistency () override;
  
  	//! Full consistency algorithm
  	__device__ void full_consistency  () override;

#endif

public:

#if CUDAON

  __device__ CudaAlldifferent ( int n_id, int n_vars, int n_args,
                     			int* vars, int* args,
                     			int* domain_idx, uint* domain_states, 
                     			int num_blocks = 1, int num_threads = 1 );
  
  __device__ virtual ~CudaAlldifferent ();
  
  
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

#endif /* defined(__NVIDIOSO__global_constraint__) */
