//
//  CudaVariable.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements the representation of a variable
//  designed for a CUDA implementation.
//

#ifndef NVIDIOSO_cuda_variable_h
#define NVIDIOSO_cuda_variable_h

#include "int_variable.h"

class CudaVariable : public IntVariable {
public:
  /**
   * Base constructor: create a variable with new id.
   * The id is given by a global id generator.
   */
  CudaVariable  ();
  
  /**
   * One parameter constructor: create a variable with a given id.
   * @param idv identifier to give to the variable
   */
  CudaVariable  ( int idv );
  ~CudaVariable ();
  
  /**
   * Set domain's bounds.
   * If no bounds are provided, an unbounded domain (int) is istantiated.
   * If an array of elements A is provided, the function instantiates a
   * domain D = [min A, max A], deleting all the elements d in D s.t.
   * d does not belong to A.
   */
  void set_domain ();
  
  /**
   * Set domain's bounds. 
   * A new domain [lw, ub] is generated.
   * @param lw lower bound
   * @param ub upper bound
   */
  void set_domain ( int lw, int ub );
  
  /**
   * Set domain's elements.
   * A domain {d_1, ..., d_n} is generated.
   * @param elems vector of vectors (subsets) of domain's elements
   * @todo implement set of sets of elements.
   */
  void set_domain ( std::vector < std::vector < int > > elems );
  
  //! print info about the current domain
  void print () const;
};

#endif

