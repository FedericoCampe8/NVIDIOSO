//
//  cuda_concrete_bitmaplist.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 17/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is the concrete implementation of the
//  cuda_concrete_domain considering list of bitmaps
//  {min, max} of (non) contiguous domain's elements.
//
//  @note A bitmap list is represented as follows:
//
//            ...|| L | U | BITMAP | L | U | BITMAP |... ,
//
//        where L/U is the lower/upper bound of the elements of
//        BITMAP when the BITMAP has been created/added to the list.
//        BITMAP has a fixed size (i.e., fixed number of chunks)
//        correspoinding to the number of chunks needed to represent
//        the largest set of consecutive values (i.e., max ( U - L + 1 )).
//
//  @note This class does not use an actual list data structure.
//        Instead, it uses a domain bitmap representation.
//        This is done in order to write C code that could be used
//        later on CUDA kernels.
//
//  @note This class do not use any auxiliary member to handle the
//        methods that will modify the iternal domain's representation
//        (i.e., pointers to each bitmap). This is done for
//        implementation reasons in order to "directly replay" the same
//        algoithms in C CUDA code.
//


#ifndef NVIDIOSO_cuda_concrete_bitmaplist_h
#define NVIDIOSO_cuda_concrete_bitmaplist_h

#include "cuda_concrete_bitmap.h"

class CudaConcreteBitmapList : public CudaConcreteDomainBitmap {
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
  
  /**
   * Given the index of a pair, return
   * the correspondend position/index in the
   * BITMAP field.
   * @param i index of the pair (lower bound L).
   * @return index of the position in BITMAP field.
   */
  int pair_to_idx ( int i ) const;
  
protected:
  
  //! Number of pairs in the list (list size).
  int _num_bitmaps;
  
  //! Fixed size of each bitmap in the list.
  int _bitmap_size;
  
  /**
   * Current domain size,
   * i.e., sum of the elements on each bitmap.
   */
  unsigned int _domain_size;
  
  /**
   * Find the index of the pair containing val.
   * @param val to be searched in the list of pairs.
   * @return the index of the pair containing val, -1 otherwise.
   * @note it returns the index of the pair regardless of 
   *       whether the element is present or not.
   */
  int find_pair ( int val ) const;
  
  /**
   * Find the index of the last pair with values smaller than val.
   * @param val to be compared in the list of pairs.
   * @return the index of the pair with val lower than val, -1 if
   *         no such pair exists.
   * @note it returns the index of the pair regardless of
   *       whether the element is present or not.
   */
  int find_prev_pair ( int val ) const;
  
  /**
   * Find the index of the first pair with values greater than val.
   * @param val to be compared in the list of pairs.
   * @return the index of the pair with val greater than val, -1 if
   *         no such pair exists.
   * @note it returns the index of the pair regardless of
   *       whether the element is present or not.
   */
  int find_next_pair ( int val ) const;
  
public:
  /**
   * Constructor.
   * It allocates size bytes for the internal domain's representation 
   * and it initializes it with the pairs of bounds contained in pairs.
   * @param size the number of bytes to allocate.
   * @param pairs the SORTED list of pairs to allocate.
   */
  CudaConcreteBitmapList ( size_t size, std::vector< std::pair <int, int> > pairs );
  
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
   * @note it is possible to add only bitmaps with empty intersection
   *       with previous bitmaps and which min is greater than current
   *       lower bound.
   * @todo complete add function to add any bitmap.
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
	 * It prints the current domain representation (its state).
   * @note it prints the content of the object given by
   *       "get_representation ()".
	 */
  void print () const;
  
};




#endif
