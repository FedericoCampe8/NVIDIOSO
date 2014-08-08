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
 * - domain increase min      ( MIN_EVT )
 * - domain decrease max      ( MAX_EVT )
 * - domain has been modified ( CHANGE_EVT )
 * - domain is empty          ( FAIL_EVT )
 */
enum class EventType {
  NO_EVT        = 0,
  SINGLETON_EVT = 1,
  BOUNDS_EVT    = 2,
  MIN_EVT       = 3,
  MAX_EVT       = 4,
  CHANGE_EVT    = 5,
  FAIL_EVT      = 6,
  OTHER_EVT     = 7
};

class Domain {
protected:
  
  //! Debug info string
  std::string _dbg;
  
  //! Domain type
  DomainType _dom_type;
  
  //! Constructor
  Domain ();
  
public:
  
  virtual ~Domain ();
  
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
  
  /**
   * Clone the current domain and returns a pointer to it.
   * @return a pointer to a domain that has been initialized as
   *         a copy (clone) of this domain.
   */
  virtual DomainPtr clone () const = 0;
  
  /**
   * Get the current event on the domain.
   * @return an event described as EventType that
   *         represents the current event (state)
   *         of this domain.
   */
  virtual EventType get_event () const = 0;
  
  /**
   * Returns the size of the domain.
   * @return the size of this domain.
   */
  virtual size_t get_size () const = 0;
  
  /**
   * Returns true if the domain is empty.
   * @return true if this domain is empty, false otherwise.
   */
  virtual bool is_empty () const = 0;
  
  /**
   * Returns true if the domain has only one element.
   * @return true if this domain is a singleton, false otherwise.
   */
  virtual bool is_singleton () const = 0;
  
  /**
   * Specifies if domain is a finite domain of numeric values (integers).
   * @return true if domain contains numeric values (not reals).
   */
  virtual bool is_numeric () const = 0;
  
  /**
   * Returns a string description of this domain, i.e., 
   * the list of values in the current domain.
   * @return a string representing the values in this domain.
   */
  virtual std::string get_string_representation () const = 0;
  
  //! Print info about this domain
  virtual void print () const = 0;
};

#endif
