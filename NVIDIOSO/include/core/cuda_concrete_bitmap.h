//
//  cuda_concrete_bitmap.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 15/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is the concrete implementation of the
//  cuda_concrete_domain considering bitmaps of values.
//

#ifndef NVIDIOSO_cuda_concrete_bitmap_h
#define NVIDIOSO_cuda_concrete_bitmap_h

#include "cuda_concrete_domain.h"

class CudaConcreteDomainBitmap : public CudaConcreteDomain { 
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
  /**
   * Macro for the size of a byte in terms of bits.
   */
  static constexpr int BITS_IN_BYTE = INT8_C( 8 );
  
  /**
   * Macro for the size of a chunk in terms of bits.
   */
  static constexpr int BITS_IN_CHUNK = sizeof( int ) * BITS_IN_BYTE;
  
  /**
   * Get index of the chunk of bits containing the bit
   * representing the value given in input.
   * @param max lower bound used to calculated the index of the bitmap
   * @return number of int used as bitmaps to represent max
   */
  static constexpr int IDX_CHUNK ( int val ) {
    return val / ( sizeof ( int ) * BITS_IN_BYTE );
  }
  
  /**
   * Get index of the bit that represents the value val module
   * the size of a chuck, i.e., the position of the corresponding bit
   * within a chunk.
   * @param val the value w.r.t. the function calculates its position
   *        within a chunk of bits
   * @return position (starting from 0) of the bit corresponding to val.
   */
  static constexpr int IDX_BIT ( int val ) {
    return val % BITS_IN_CHUNK;
  }
  
  /**
   * Get the number of chunks needed to represent a domain of size values.
   * @param size the size in terms of number of elements of the domain to 
   *        represent as bitmap.
   * @return number of chunks needed to represent size valus.
   */
  static constexpr int NUM_CHUNKS ( int size ) {
    return size % BITS_IN_CHUNK == 0 ?
    (size / BITS_IN_CHUNK == 0 ? 1 : size / BITS_IN_CHUNK) :
    (size / BITS_IN_CHUNK + 1);
  }
  
  //! Number of bits set to 1
  unsigned int _num_valid_bits;
  
public:
  /**
   * Constructor for CudaConcreteDomainBitmap.
   * @param size the size in bytes to allocate for the bitmap.
   * @note the bitmap is represented
   *       considering lower bound = 0 and upper bound 
   *       given by the parameter size.
   * @note initially all bits are set to 1 (i.e. valid bits).
   */
  CudaConcreteDomainBitmap ( size_t size );
  
  /**
   * Constructor for CudaConcreteDomainBitmap.
   * @param size the size in bytes to allocate for the bitmap.
   * @param min lower bound for {min, max} set initilization.
   *        min must be greater than or equal to 0 and
   *        less than or equal to the max number of bits storable
   *        using size bytes.
   * @param max upper bound for {min, max} set initilization.
   *        max must be less than or equal to max number of bits 
   *        storable using size bytes and greater than or equal to 0.
   * @note the bitmap is represented
   *       considering lower bound = 0 and upper bound
   *       given by the parameter size.
   * @note initially all bits in {min, max} are set to 1 (i.e. valid bits).
   */
  CudaConcreteDomainBitmap ( size_t size, int min, int max );
  
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
   * @note  value is given w.r.t. a lower bound of 0.
	 */
  void add ( int value );
  
  /**
	 * It computes union of this domain and {min, max}.
	 * @param min lower bound of the new domain which is being added.
   * @param max upper bound of the new domain which is being added.
   * @todo implement using checks on chunks of bits (i.e. sublinear cost).
	 */
  void add ( int min, int max );
  
  /**
   * It checks whether the value belongs to
   * the domain or not.
   * @param value to check whether it is in the current domain.
   * @note value is given w.r.t. the lower bound of 0.
   */
  bool contains ( int value ) const;
  
  /**
   * It checks whether the current domain contains only
   * an element (i.e., it is a singleton).
   * @return true if the current domain is singleton,
   *         false otherwise.
   */
   bool is_singleton () const;
  
  /**
   * It returns the value of the domain element if it is a singleton.
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
   *       "get_representation ()".
	 */
   void print () const;
};

#endif
