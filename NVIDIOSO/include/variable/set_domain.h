//
//  set_domain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements an abstract class for a "set" domain,
//  i.e., a domain whose values are specified by a set
//  {x_1, ..., x_n} of elements.
//

#ifndef NVIDIOSO_set_domain_h
#define NVIDIOSO_set_domain_h

#include "domain.h"

class SetDomain : public Domain {
protected:
  // Set of elements
  std::vector< int > _d_elements;
  
  DomainPtr clone_impl () const;
  
public:
  SetDomain ();
  virtual ~SetDomain ();
  
  /**
   * Set bounds and perform some consistency checking.
   * It throws "no solutions" if consistency checking fails.
   * @param elems vector of domain's elements
   */
  virtual void set_values ( std::vector< int > elems );
  
  /**
   * Get a vector containing the current values
   * contained in the domain.
   * @return the current elements in the domain
   */
  virtual std::vector< int > get_values () const;
  
  //! Clone the current domain and returns a pointer to it
  DomainPtr clone () const;
  
  /**
   * Get event on this domain
   * @todo implement this function
   */
  EventType get_event () const;
  
  /**
   * Sets the no event on this domain.
   * @note No event won't trigger
   *       any propagation on this domain.
   */
  void reset_event ();
  
  //! Returns the size of the domain
  size_t get_size () const;
  
  //! Returns true if the domain is empty
  bool is_empty () const;
  
  //! Returns true if the domain has only one element
  bool is_singleton () const;
  
  //! Returns true if this is a numeric finite domain
  bool is_numeric () const;
  
  //! Get string rep. of this domain
  std::string get_string_representation () const;
  
  //! Print info about the domain
  void print () const;
};

#endif
