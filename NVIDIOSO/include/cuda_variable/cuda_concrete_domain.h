//
//  CudaConcreteDomain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 15/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class is an abstract class that implements the interface
//  of a ConcreteDomain for a CUDA implementation.
//  It specializes the concrete domain on integers.
//  Concrete domain is represented by an array of bits (i.e., bitmap).
//

#ifndef NVIDIOSO_cuda_concrete_domain_h
#define NVIDIOSO_cuda_concrete_domain_h

#include "concrete_domain.h"

class CudaConcreteDomain;
typedef std::shared_ptr< CudaConcreteDomain > CudaConcreteDomainPtr;

class CudaConcreteDomain : public ConcreteDomain<int> {
protected:
  std::string _dbg;
  
  //! Number of allocated (32 bit int) chunks
  int _num_chunks;
  
  //! Lower bound
  int _lower_bound;
  
  //! Upper bound
  int _upper_bound;
  
  /**
   * Concrete domain is represented by an array of (32 bit) integers.
   * @note actual internal representation of domain.
   */
  int * _concrete_domain;
  
  /**
   * Flush domain: reduces its domain size to zero by
   * flushing all values in the internal domain's representation.
   * It sets the current domain's state as empty.
   * @note it sets upper bound < lower bound.
   */
  void flush_domain ();
  
  /**
   * Empty domain: reduces its domain size to zero by
   * setting the current domain's state as empty.
   * @note it does not flush the current internal 
   * domain's representation.
   */
  void set_empty ();
  
  /**
   * Constructor for CudaConcreteDomain.
   * It instantiates a new object and
   * allocate size bytes for the array of integers
   * @param size the number of bytes to allocate.
   * @note the client should check whether integers
   *       are represented by 32 bit values.
   */
  CudaConcreteDomain ( size_t size );
  
public:
  virtual ~CudaConcreteDomain ();
  
  //! Returns lower bound
  int lower_bound () const override;
  
  //! Returns upper bound
  int upper_bound () const override;
  
  /**
   * Get the number of allocated
   * chunks (in terms of 32 bit integers).
   */
   int get_num_chunks () const;
  
  /**
   * Get the number of allocated
   * bytes, i.e., the size of the internal
   * domain's representation.
   */
   size_t allocated_bytes () const;
  
  /**
  * It checks whether the current domain is empty.
   * @return true if the current domain is empty,
   *         false otherwise.
   */
   bool is_empty () const;
  
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
  
  /**
	 * It returns a void pointer to an object representing the
   * current representation of the domain (e.g., bitmap).
	 * @return void pointer to the concrete domain representation.
	 */
   const void * get_representation () const override;
  
  /**
   * Returns the current CUDA concrete domain's representation.
   * @return an integer id indicating the current representation of 
   *         this domain.
   */
  virtual int get_id_representation () const = 0;
};


#endif
