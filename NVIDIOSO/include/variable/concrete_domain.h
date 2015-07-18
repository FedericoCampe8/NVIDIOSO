//
//  concrete_domain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 14/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for the concrete representation
//  of a domain.
//  A concrete representation of a domain can be done with lists,
//  maps, arrays or other data structures.
//

#ifndef NVIDIOSO_concrete_domain_h
#define NVIDIOSO_concrete_domain_h

#include "globals.h"

template< class T >
class ConcreteDomain {
public:
  
  /**
   * It returns the number of elements in the domain.
   * It returns the current size of the domain.
   */
  virtual unsigned int size () const = 0;
  
  //! Returns lower bound
  virtual T lower_bound () const = 0;
  
  //! Returns upper bound
  virtual T upper_bound () const = 0;
  
  /**
	 * It updates the domain to have values only within min/max.
	 * @param min new lower bound to set for the current domain.
	 * @param max new upper bound to set for the current domain.
	 */
  virtual void shrink ( T min, T max ) = 0;
  
  /**
	 * It substracts {value} from the current domain.
	 * @param value the value to subtract from the current domain.
	 */
  virtual void subtract ( T value ) = 0;
  
  /**
   * It updates the domain according to the minimum value.
   * @param min domain value.
   */
  virtual void in_min ( T min ) = 0;
  
  /**
   * It updates the domain according to the maximum value.
   * @param max domain value.
   */
  virtual void in_max ( T max ) = 0;
  
  /**
	 * It computes union of this domain and {value}.
	 * @param value it specifies the value which is being added.
	 */
  virtual void add ( T value ) = 0;
  
  /**
	 * It computes union of this domain and {min, max}.
	 * @param min lower bound of the new domain which is being added.
   * @param max upper bound of the new domain which is being added.
	 */
  virtual void add ( T min, T max ) = 0;
  
  /**
   * It checks whether the value belongs to 
   * the domain or not.
   * @param value to check whether it is in the current domain.
   */
  virtual bool contains ( T value ) const = 0;
  
  /**
   * It checks whether the current domain is empty.
   * @return true if the current domain is empty,
   *         false otherwise.
   */
  virtual bool is_empty () const = 0;
  
  /**
   * It checks whether the current domain contains only 
   * an element (i.e., it is a singleton).
   * @return true if the current domain is singleton,
   *         false otherwise.
   */
  virtual bool is_singleton () const = 0;
  
  /**
   * It returns the value of type T of the domain
   * if it is a singleton.
   * @return the value of the singleton element.
   * @note Classes that specialize this method
   *       should handle the case of an invokation
   *       of the method and a non-singleton domain.
   *       For example, throw an exception or returning
   *       the lower bound.
   */
  virtual T get_singleton () const = 0;
  
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
  virtual void set_domain ( void * const domain,
                            int rep, int min, int max, int dsz ) = 0;
  
  /**
	 * It returns a void pointer to an object representing the 
   * current representation of the domain (e.g., bitmap).
	 * @return void pointer to the concrete domain representation.
	 */
  virtual const void * get_representation () const = 0;
  
  /**
	 * It prints the current domain representation (its state).
   * @note it prints the content of the object given by
   *       "get_representation ()" .
	 */
  virtual void print () const = 0;
};

#endif




