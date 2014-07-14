//
//  cp_domain.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class defines a domain and related operations on it.
//  It represents the common interface for every domain that can be
//  used as FD domain for the variables.
//  It can be specilized for any kind of domain (e.g., CUDA-based domains).
//

#ifndef NVIDIOSO_domain_h
#define NVIDIOSO_domain_h

#include "globals.h"

class Domain;
typedef std::shared_ptr<Domain> DomainPtr;

/**
 * Type of domain.
 * - Boolean ( BOOLEAN ),
 * - Integer ( INTEGER ),
 *  -Set     ( SET ).
 * Other types may be added in future.
 */
enum class DomainType {
  BOOLEAN,
  INTEGER,
  SET,
  OTHER
};

/**
 * Events that happen on the domain.
 * These events trigger specific propagators:
 * - no event as occured      ( NO_EVT )
 * - domain is singleton      ( SINGLETON_EVT )
 * - domain changed bounds    ( BOUNDS_EVT )
 * - domain has been modified ( CHANGE_EVT )
 * - domain is empty          ( FAIL_EVT )
 */
enum class EventType {
  NO_EVT,
  SINGLETON_EVT,
  BOUNDS_EVT,
  CHANGE_EVT,
  FAIL_EVT,
  OTHER_EVT
};

class Domain {
protected:
  
  // Info
  std::string _dbg;
  
  // Domain type
  DomainType _dom_type;
  
public:
  
  Domain ();
  virtual ~Domain ();
  
  // Get/set methods
  //! Constants for int min/max domain bounds
  static constexpr int MIN_DOMAIN () { return INT32_MIN; }
  //! Constants for int min/max domain bounds
  static constexpr int MAX_DOMAIN () { return INT32_MAX; }
  
  /**
   * Set domain's type (use get_type to get the type).
   * @param dt domain type of type DomainType
   */
  void       set_type ( DomainType dt );
  DomainType get_type () const;
  
  //! Clone the current domain and returns a pointer to it
  virtual DomainPtr clone () const = 0;
  
  //! Get the current event on the domain
  virtual EventType get_event () const = 0;
  
  //! Returns the size of the domain
  virtual size_t get_size () const = 0;
  
  //! Returns true if the domain is empty
  virtual bool is_empty () const = 0;
  
  //! Returns true if the domain has only one element
  virtual bool is_singleton () const = 0;
  
  //! Print info about the current domain
  virtual void print () const = 0;
};

#endif
