//
//  cuda_concrete_list.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 15/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is the concrete implementation of the
//  cuda_concrete_domain considering list of pairs
//  {min, max} of contiguous domain's elements.
//  @note This class does not use an actual list data structure.
//        Instead, it uses a domain bitmap representation.
//        This is done in order to write C code that could be used
//        later on CUDA kernels.
//

#ifndef NVIDIOSO_cuda_concrete_list_h
#define NVIDIOSO_cuda_concrete_list_h

#include "cuda_concrete_domain.h"

class CudaConcreteDomainList : public CudaConcreteDomain {
private:
  /**
   * Initial lower bound
   * @note this bound is used to check consistecy on
   *       actions that modify domain's size.
   */
  int _init_lower_bound;
  
  /**
   * Initial upper bound
   * @note this bound is used to check consistecy on
   *       actions that modify domain's size.
   */
  int _init_upper_bound;
  
protected:
  
  //! Number of pairs in the list (list size)
  int _num_pairs;
  
  //! Max number of storable pairs in the concrete domain
  int _max_allowed_pairs;
  
  /**
   * Current domain size,
   * i.e., sum of the elements on each 
   * pair of bounds in the list.
   */
  unsigned int _domain_size;
  
  /**
   * Find the index of the pair containing val.
   * @param val to be searched in the list of pairs.
   * @return the index of the pair containing val, -1 otherwise.
   */
  int find_pair ( int val ) const;
  
  /**
   * Find the index of the last pair with values smaller than val.
   * @param val to be compared in the list of pairs.
   * @return the index of the pair with val lower than val, -1 if 
   *         no such pair exists.
   */
  int find_prev_pair ( int val ) const;
  
  /**
   * Find the index of the first pair with values greater than val.
   * @param val to be compared in the list of pairs.
   * @return the index of the pair with val greater than val, -1 if
   *         no such pair exists.
   */
  int find_next_pair ( int val ) const;
  
public:
  
  /**
   * Constructor for CudaConcreteDomainList.
   * @param size the size in bytes to allocate for the bitmap.
   * @param min lower bound in {min, max}
   * @param max upper bound in {min, max}
   */
  CudaConcreteDomainList ( size_t size, int min, int max );
  
  //! It returns the current size of the domain.
  unsigned int size () const;
  
  /**
	 * It updates the domain to have values only within min/max.
	 * @param min new lower bound to set for the current domain.
	 * @param max new upper bound to set for the current domain.
	 */
  void shrink ( int min, int max );
  
  /**
	 * It substracts {value} from the current domain.
	 * @param value the value to subtract from the current domain.
   * @note a value is removed only if it corresponds to a lower/upper bound.
	 */
  void subtract ( int value );
  
  /**
   * It updates the domain according to min value.
   * @param min domain value.
   */
  void in_min ( int min );
  
  /**
   * It updates the domain according to max value.
   * @param max domain value.
   */
  void in_max ( int max );
  
  /**
	 * It computes union of this domain and {value}.
	 * @param value it specifies the value which is being added.
	 */
  void add ( int value );
  
  /**
	 * It computes union of this domain and {min, max}.
	 * @param min lower bound of the new domain which is being added.
   * @param max upper bound of the new domain which is being added.
	 */
  void add ( int min, int max );
  
  /**
   * It checks whether the value belongs to
   * the domain or not.
   * @param val to check whether it is in the current domain.
   * @note val is given w.r.t. the lower bound of 0.
   */
  bool contains ( int val ) const;
  
  /**
   * It checks whether the current domain contains only
   * an element (i.e., it is a singleton).
   * @return true if the current domain is singleton,
   *         false otherwise.
   */
  bool is_singleton () const;
  
  /**
   * It returns the value of type T of the domain
   * if it is a singleton.
   * @return the value of the singleton element.
   * @note it throws an exception if domain is not singleton.
   */
  int get_singleton () const;
  
  /**
	 * It returns a void pointer to an object representing the
   * current representation of the domain (e.g., bitmap).
	 * @return void pointer to the concrete domain representation.
	 */
  const void * get_representation () const;
  
  /**
	 * It prints the current domain representation (its state).
   * @note it prints the content of the object given by
   *       "get_representation ()" .
	 */
  void print () const;
};



#endif
