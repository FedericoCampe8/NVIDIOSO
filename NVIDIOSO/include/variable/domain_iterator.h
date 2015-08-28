//
//  domain_iterator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Iterator's interface that allows the client to
//  explore the domain of a given variable.
//

#ifndef __NVIDIOSO__domain_iterator__
#define __NVIDIOSO__domain_iterator__

#include "globals.h"
#include "int_domain.h"

class DomainIterator {
protected:
  IntDomainPtr _domain;
  
public:
  DomainIterator ( IntDomainPtr domain );
  
  virtual ~DomainIterator();
  
  /**
   * Checks if the current domain is a numeric domain.
   * @return true if current domain is numeric 
   *         (i.e., int domain), false otherwise.
   */
  virtual bool is_numeric () const;
  
  /**
   * Checks if the current domain contains the given value.
   * @param val integer value to look.
   * @return true if current domain constains val, false otherwise.
   */
  virtual bool contains ( int val ) const;
  
  /**
   * Returns the current minimal value in domain.
   * @return the minimum value belonging to the domain.
   */
  virtual int min_val () const;
  
  /**
   * Returns the current maximal value in domain.
   * @return the maximum value belonging to the domain.
   */
  virtual int max_val () const;
  
  /**
   * Returns a random value from domain.
   * @return the a random value belonging to the domain.
   */
  virtual int random_val () const;
  
  /**
   * Returns the current domain's size.
   * @return current domain's size.
   */
  virtual size_t domain_size () const;
  
  /**
   * Set a concrete domain.
   * It overrides the current concrete domain representation.
   * @note the client must provide a consistent internal domain's representation.
   */
  virtual void set_domain_status ( void * concrete_domain );
  
  /**
   * Returns a pointer to an area of memory storing
   * the current internal domain.
   * @return a pair containing: 
   *         (1) the size of the current internal domain's representation; 
   *         (2) a void pointer to an area of memory
   *         where the current internal representation of the domain
   *         associated to this iterator is store.
   */
  virtual std::pair<size_t, const void *> get_domain_status () const;
  
  /**
   * Returns a string description of this domain, i.e.,
   * the list of values in the current domain.
   * @return a string representing the values in this domain.
   */
  virtual std::string get_string_representation () const;
};

#endif /* defined(__NVIDIOSO__domain_iterator__) */
