//
//  cuda_domain.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/09/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#ifndef NVIDIOSO_cuda_domain_h
#define NVIDIOSO_cuda_domain_h

#include "int_domain.h"
#include "cuda_concrete_domain.h"

enum class CudaDomainRepresenation {
  BITMAP,
  BITMAP_LIST,
  LIST,
  OTHER
};

class CudaDomain : public IntDomain {
protected:
  // List representation base case (internal state)
  static constexpr int INT_BITMAP  =  0;
  static constexpr int INT_BITLIST = -1;
  static constexpr int INT_LIST    =  1;
  
  /**
   * Constants used to retrieve the current domain description.
   * Domain represented as:
   *  | EVT | REP | LB | UB | DSZ || ... BIT ... |.
   * See system_description.h.
   */
  static constexpr int EVT_IDX () { return 0; }
  static constexpr int REP_IDX () { return 1; }
  static constexpr int  LB_IDX () { return 2; }
  static constexpr int  UB_IDX () { return 3; }
  static constexpr int DSZ_IDX () { return 4; }
  static constexpr int BIT_IDX () { return 5; }
  
  /**
   * Macro to use for declaring the
   * size of a byte in terms of bits.
   */
  static constexpr int BITS_IN_BYTE = INT8_C( 8 );
  
  /**
   * Shared memory available.
   * @note keep 1 kB less than the actual memory available.
   */
  static constexpr int SHARED_MEM_KB = 47;
  
  /**
   * Maximum domain size in terms of bytes.
   * @note  see CUDA specifications.
   *        Usually,
   *        (48 - 1) kB =
   *         47 * 1024  = 48128 Byte.
   */
  static constexpr size_t MAX_BYTES_SIZE = SHARED_MEM_KB * 1024;
  
  /**
   * Number of Bytes needed for representing the 
   * current domain status.
   */
  static constexpr size_t MAX_STATUS_SIZE = 5 * sizeof( int );
  
  /**
   * Maximum size in terms of storable values.
   * Worst case: list of type {1, 1}, {3, 3}, {5, 5}, ...
   * Number of integers = 
   * ((MAX_BYTES_SIZE - 5 * sizeof( int )) / sizeof( int ))
   * @note  see CUDA specifications.
   */
  static constexpr size_t MAX_DOMAIN_VALUES = ((MAX_BYTES_SIZE - MAX_STATUS_SIZE) / sizeof( int ));
  
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
   * Get index of the last int used as bitmap to
   * represent [min, max].
   * @param max lower bound used to calculated the index of the bitmap
   * @return number of int used as bitmaps to represent max
   */
  static constexpr int IDX_BIT ( int val ) {
    return val % ( sizeof ( int ) * BITS_IN_BYTE );
  }
  
  /**
   * Return the number of 32-bit integers needed to 
   * represent a set of n domain's values.
   * @param n number of values to represent as bits
   * @return number of 32-bit integer chunks needed to 
   *         represent n values.
   */
  inline static int num_chunks ( int n ) {
    return ceil ( (n * 1.0) / (BITS_IN_BYTE * sizeof( int )) );
  }
  
  /**
   * Actual domain is represented by an object of type
   * "cuda_concrete_domain".
   * This domain can be a either bitmap, a list of bounds, or 
   * a bitmap list, depending on the size of the domain.
   * Internal switches between domain representations are 
   * performed automatically as soon as the domain's size
   * is reduced to a given threshold.
   * @note system_description.h
   */
  CudaConcreteDomainPtr _concrete_domain;
  
  /**
   * Domain is the actual bit domain representation.
   * Operations are performed on _concrete_domain, status
   * is stored on _domain.
   * When another class needs this domain's representation,
   * _domain will be returned.
   */
  int * _domain;
  
  /**
   * Total allocated bytes for representing
   * the current domain.
   */
  size_t _num_allocated_bytes;
  /**
   * Total number of bitchunks.
   * @note it does not consider the first 
   *       part related to information about domain.
   */
  size_t _num_int_chunks;
  
  //! Clone method to clone the current object
  DomainPtr clone_impl () const;
  
  //! Convert the current event int to a domain event
  EventType int_to_event () const;
  
  //! Convert a domain event to the current integer
  void event_to_int ( EventType evt ) const ;
  
  /**
   * Switch to bit representation of domain.
   * @ It changes only identifier in the REP field.
   */
  void set_bit_representation ();
  
  /**
   * Switch to bitlist representation of domain.
   * @param num_list the number (positive) of bitlists.
   * @ It changes only identifier in the REP field.
   */
  void set_bitlist_representation ( int num_list = INT_BITLIST );
  
  /**
   * Switch to list representation of domain.
   * @param num_list the number (positive) of bitlists.
   * @ It changes only identifier in the REP field.
   */
  void set_list_representation ( int num_list = INT_LIST );
  
  //! Get domain representation (i.e., bitmap, bitmaplist, or list)
  CudaDomainRepresenation get_representation () const;
  
  /**
   * Take the current list representation and switch it to 
   * a bitmap list represenatation.
   * @note it doesn't work from bitmap to bitmap list.
   */
  void switch_list_to_bitmaplist ();
  
public:
  CudaDomain  ();
  
  virtual ~CudaDomain ();
  
  DomainPtr clone () const;
  
  /**
   * Initializes domain with default values:
   * - Event: no event;
   * - Representation: list or bitmap according to [min, max];
   * - Lower bound: min;
   * - Upper bound: max;
   * - Size: |max - min + 1| or MAX_INT if
   *         max = MAN_INT()/2 and min = MIN_INT() / 2, etc..
   * @note It instantiate an array of ints of at most MAX_BYTES_SIZE.
   * @param min lower bound of the domain
   * @param max upper bound of the domain
   * @return it fails whenever consistency check on
   *         min/max fails (i.e., max < min).
   */
  void init_domain ( int min, int max );
  
  /**
   * Get the number of allocated bytes needed for
   * representing the current domain w.r.t. its 
   * lower and upper bounds.
   * @return the number of allocated bytes.
   */
  size_t allocated_bytes () const;
  
  //! Get event on the current domain
  EventType get_event () const;

  /**
   * Sets the no event on this domain.
   * @note No event won't trigger
   *       any propagation on this domain.
   */
  void reset_event ();
  
  /**
   * Set a concrete domain.
   * It overrides the current concrete domain representation.
   * @note the client must provide a consistent internal domain's representation.
   */
  void set_domain_status ( void * concrete_domain );
  
  /**
   * Get the size if the current domain (internal representation).
   * @return number of bytes of the internal domain representaion.
   */
   size_t get_domain_size () const;
  
  /**
   * Get a pointer to the area of memory representing
   * the current internal representation of this domain.
   * @return const void pointer to the current domain
             (internal representation)
   */
  const void * get_domain_status () const;
  
  /**
   * Gets a reference to the current internal representation.
   * @return a reference to a (cuda) concrete domain.
   */
  const int * get_concrete_domain () const;
  
  /**
   * Get domain size.
   * It returns the currenst size of the domain, 
   * checking whether there are "holes" according to 
   * the current representation of the domain (i.e., 
   * bitmap or list):
   * @return the current domain's size.
   */
   size_t get_size () const;

  //! Get the domain's lower bound
  int lower_bound () const;
  
  //! Get the domain's upper bound
  int upper_bound () const;
  
  /**
   * It checks whether the value belongs to
   * the domain or not.
   * @param value to check whether it is in the current domain.
   * @return true if value is in this domain, false othewise
   */
  bool contains ( int value ) const;
  
  /**
   * The same as set_bounds.
   * It shrinks the domain to {min, max}.
   * @param min lower bound
   * @param max upper bound
   */
  void set_bounds ( int min, int max );
  
  /**
   * It specializes the parent method in order to
   * set up the array of (int) values.
   * It istantiates a domain [min, max].
   * This actually updates the bounds and it performs
   * consistency checking and updating of the domain size.
   * @param min lower bound
   * @param max upper bound
   */
  void shrink ( int min, int max );
  
  /**
   * Set domain as singleton as {val}.
   * @param val the value to set as singleton.
   */
  bool set_singleton ( int val );
  
  //! Subtract the element from the domain (see int_domain.h)
  bool subtract ( int n );
  
  /** 
   * Add an element val to the current domain (see int_domain.h).
   * @note if the element is out of the current bounds,
   *       no element will be added, i.e., the domain
   *       mantains the current size.
   */
  void add_element ( int n );
  
  //! Increase the lower_bound to min (see int_domain.h)
  void in_min ( int min );
  
  //! Decrease the upper_bound to max (see int_domain.h)
  void in_max ( int max );
  
  //! Print info about domain
  void print () const;
  
  //! Print internal domain representation
  void print_domain () const;
};

#endif
