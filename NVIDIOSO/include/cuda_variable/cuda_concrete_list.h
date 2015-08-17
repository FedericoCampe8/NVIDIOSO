//
//  cuda_concrete_list.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/15/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class is the concrete implementation of the
//  cuda_concrete_domain considering list of pairs
//  {min, max} of contiguous domain's elements.
//  @note This class does not use a list data structure.
//        Instead, it uses an array of bits read as pairs of bounds.
//        This is done in order to write C code that could be used
//        later on CUDA kernels.
//  @note The number of pairs actually stored into the bitmap list is saved
//        into the "_num_pairs" member. 
//        _concrete_domain will store _num_pairs pairs of bounds at positions
//        2*i and 2*i + 1, where 0 <= i < _num_pairs.
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
  std::size_t _domain_size;
  
  /**
   * Get number of elements within a given pair of bounds.
   * @param idx index of the pair of bounds to consider.
   * @return number of elements in the between the pair of bounds.
   */
  int get_elements_in_bound ( int idx );
  
  //! Return number of elements in [min, max]
  size_t get_size_from_values ( int min, int max );
   
  /**
   * Get size of domain calculated as sum of elements between each
   * pair of bounds in the range (lower_pair, upper_pair).
   * @param lower_pair index of the pair from where to start counting
   * @param upper_pair index of the pair where to stop counting
   * @return number of elements between the two pairs of bounds.
   * @note The number of elements returned by this function represents
   *       a number of elements which are not currently contained in this domain.
   */
   size_t get_size_from_bounds_btw ( int lower_pair, int upper_pair );
   
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
  
  /**
   * Sets the internal representation of the domain
   * from a given concrete domain and given lower/upper bounds.
   * @param domain a reference to a given concrete domain.
   * @param rep current internal's domain representation.
   * @param min lower bound to set.
   * @param max upper bound to set.
   * @param dsz domain size to set.
   * @note the client must pass a valid concrete domain's representation.
   */
  void set_domain ( void * const domain,
                    int rep, int min, int max, int dsz ) override;
  
  //! It returns the current size of the domain.
  unsigned int size () const override;
  
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
   * Returns the current CUDA concrete domain's representation.
   * @return an integer id indicating the current representation of
   *         this domain.
   */
  int get_id_representation () const override;
  
  /**
	 * It prints the current domain representation (its state).
   * @note it prints the content of the object given by
   *       "get_representation ()" .
	 */
  void print () const;
};



#endif
