//
//  BoolDomain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class implements an abstract class for
//  an "Boolean" domain, i.e., a domain with only 2 values,
//  namely, True and False.
//  Current implementation: bool domain tracted as int domain with 2 values.
//  @todo: implement this class.
//

#ifndef NVIDIOSO_bool_domain_h
#define NVIDIOSO_bool_domain_h

#include "domain.h"

enum class BoolValue {
  TRUE_VALUE,
  FALSE_VALUE,
  UNDEF_VALUE,
  EMPTY_VALUE
};

class BoolDomain : public Domain {
protected:
  
  //! Current domain value
  BoolValue _bool_value;
  
  //! Clone the current domain
  DomainPtr clone_impl () const;
  
public:
  
  BoolDomain ();
  virtual ~BoolDomain ();
  
  //! Clone the current domain and returns a pointer to it
  DomainPtr clone () const;
  
  /**
   * Get event on this domain
   * @todo implement this function
   */
  EventType get_event () const;
  
  //! Returns the size of the domain
  size_t get_size () const;
  
  //! Returns true if the domain is empty
  bool is_empty () const;
  
  //! Returns true if the domain has only one element
  bool is_singleton () const;
  
  //! Print info about the domain
  void print () const;
};


#endif
