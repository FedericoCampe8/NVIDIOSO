//
//  int_variable.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 29/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class specializes the FD Variable on integers and it is an
//  interface for concrete classes on integers.
//

#ifndef __NVIDIOSO__int_variable__
#define __NVIDIOSO__int_variable__

#include "variable.h"
#include "int_domain.h"
#include "backtrackable_object.h"

class IntVariable;
typedef std::shared_ptr<IntVariable> IntVariablePtr;

class IntVariable : public Variable, public BacktrackableObject< std::vector<int> > {
protected:
  /**
   * Pointer to the domain of the variable.
   * IntDomain for IntVariable
   */
  IntDomainPtr _domain_ptr;
  
  IntVariable ();
  IntVariable ( int idv );

public:
  
  /**
   * Set domain's bounds.
   * If no bounds are provided, an unbounded domain (int) is istantiated.
   * If an array of elements A is provided, the function instantiates a
   * domain D = [min A, max A], deleting all the elements d in D s.t.
   * d does not belong to A.
   */
  virtual void set_domain () = 0;
  
  /**
   * Set domain's bounds.
   * A new domain [lw, ub] is generated.
   * @param lw lower bound
   * @param ub upper bound
   */
  virtual void set_domain ( int lw, int ub ) = 0;
  
  /**
   * Set domain's elements.
   * A domain {d_1, ..., d_n} is generated.
   * @param elems vector of vectors (subsets) of domain's elements
   * @todo implement set of sets of elements.
   */
  virtual void set_domain ( std::vector < std::vector < int > > elems ) = 0;
  
  //! Get event on this domain
  virtual EventType get_event () const;
  
  /**
   * Set domain according to the specific
   * variable implementation.
   * @note: different types of variable
   * @param dt domain type of type DomainType to set
   *        to the current variable
   */
  void set_domain_type ( DomainType dt );
  
  /**
   * It returns the size of the current domain.
   * @return the size of the current variable's domain.
   */
  size_t get_size () const;
  
  /**
   * It checks if the domain contains only one value.
   * @return true if the the variable's domain is a singleton,
   *         false otherwise.
   */
  bool is_singleton () const;
  
  /**
   * It checks if the domain is empty.
   * @return true if variable domain is empty.
   *         false otherwise.
   */
  bool is_empty () const;
  
  /**
   * It returns the current minimal value in the
   * domain of this variable.
   * @return the minimum value belonging to the domain.
   * @note the same value can be obtained by using the 
   *       domain iterator. 
   */
  virtual int min() const;
  
  /**
   * It returns the current maximal value in the
   * domain of this variable.
   * @return the maximum value belonging to the domain.
   * @note the same value can be obtained by using the
   *       domain iterator.
   */
  virtual int max() const;
  
  /**
   * Set domain's bounds.
   * It updates the domain to have values only within the
   * interval min..max.
   * @note it does not update _lower_bound and
   *       _upper_bound here for efficiency reasons.
   * @param lower lower bound value
   * @param upper upper bound value
   */
  virtual void shrink ( int min, int max );
  
  /**
   * It intersects with the domain which is a complement
   * of the value given as input, i.e., subtract a value from
   * the current domain.
   * @param val the value to subtract from the current domain
   * @return true if succeed, false otherwise.
   */
  virtual bool subtract ( int val );
  
  /**
   * It updates the domain according to the minimum value.
   * @param min domain value.
   */
  virtual void in_min ( int min );
  
  /**
   * It updates the domain according to the maximum value.
   * @param max domain value.
   */
  virtual void in_max ( int max );
  
  //! print info about the current domain
  virtual void print () const;
};

#endif /* defined(__NVIDIOSO__int_variable__) */
