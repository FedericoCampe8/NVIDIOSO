//
//  int_domain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements an abstract class for
//  an "int" domain, i.e., a domain with continguous
//  elements within the range lower bound / upper bound.
//
//  @note: Pure methods:
//  to be defined by the derived classes, according to the
//  domain representation chosen for the concrete implementation
//  of a IntDomain class.
//  @note: need also to specialize the following method (from Domain):
//         DomainPtr clone () const;
//


#ifndef NVIDIOSO_int_domain_h
#define NVIDIOSO_int_domain_h

#include "domain.h"

class IntDomain;
typedef std::shared_ptr<IntDomain> IntDomainPtr;

class IntDomain : public Domain {
protected:
  IntDomain ();
  
public:
  virtual ~IntDomain ();
  
  //! Returns true if the domain has only one element
  bool is_singleton () const;
  
  //! Returns true if the domain is empty
  bool is_empty () const;
  
  //! Returns true if this is a numeric finite domain
  bool is_numeric () const;
  
  //! Get string rep. of this domain
  std::string get_string_representation () const;
  
  /**
   * Set a concrete domain.
   * It overrides the current concrete domain representation.
   * @note the client must provide a consistent internal domain's representation.
   */
  virtual void set_domain_status ( void * domain );
  
  /**
   * Get the size if the current domain (internal representation).
   * @return number of bytes of the internal domain representaion.
   * @note default is 0.
   */ 
  virtual size_t get_domain_size () const; 
  
  /**
   * Get a pointer to the area of memory representing
   * the current internal representation of this domain.
   * @return const void pointer to the current domain
   (internal representation)
   * @note default is nullptr
   */
  virtual const void * get_domain_status () const;
  
  //! Print base info about int domain
  virtual void print () const;
  
  //! Get the domain's lower bound
  virtual int lower_bound () const = 0;
  
  //! Get the domain's upper bound
  virtual int upper_bound () const = 0;
  
  /**
   * It checks whether the value belongs to
   * the domain or not.
   * @param value to check whether it is in the current domain.
   * @return true if value is in this domain, false othewise
   */
  virtual bool contains ( int value ) const = 0;
  
  /**
   * Initialize domain:
   * this function is used to set up the domain 
   * as soon it is created.
   * Classes that derive IntDomain specilize this 
   * method according to their internal representation
   * of domain.
   */
  virtual void init_domain ( int min, int max ) = 0;
  
  /**
   * Set domain's bounds.
   * It updates the domain to have values only within the
   * interval min..max.
   * @param lower lower bound value
   * @param upper upper bound value
   */
  virtual void shrink ( int min, int max ) = 0;
  
  /**
   * Set domain to the singleton element
   * given in input.
   * @param val the value to set as singleton
   * @return true if the domain has been set to
   *         singleton, false otherwise.
   */
  virtual bool set_singleton ( int val ) = 0;
  
  /**
   * It intersects with the domain which is a complement
   * of the value given as input, i.e., subtract a value from
   * the current domain.
   * @param val the value to subtract from the current domain
   * @return true if succeed, false otherwise.
   */
  virtual bool subtract ( int val ) = 0;
  
  /**
   * It computes the union of the current domain
   * with the domain represented by the singleton element
   * given in input to the method.
   * If the element is out of  [lower_bound, upper_bound]
   * it enlarges the domain.
   * @param val element to add to the current domain.
   */
  virtual void add_element ( int val ) = 0;
  
  /**
   * It updates the domain according to the minimum value.
   * @param min domain value.
   */
  virtual void in_min ( int min ) = 0;
  
  /**
   * It updates the domain according to the maximum value.
   * @param max domain value.
   */
  virtual void in_max ( int max ) = 0;
};

#endif


