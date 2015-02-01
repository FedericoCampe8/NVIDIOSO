//
//  cp_constraint_store.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  This class represents the interface for a Constraint Store.
//  It holds the information about the state of the process related
//  to the constraints and the constraints to be propagated.
//  Any Constraint Store specific for an application/hardware (e.g., CUDA)
//  must implement the method specified below.
//

#ifndef NVIDIOSO_constraint_store_h
#define NVIDIOSO_constraint_store_h

#include "globals.h"
#include "constraint.h"

class ConstraintStore;
typedef std::shared_ptr<ConstraintStore> ConstraintStorePtr;

class ConstraintStore {
public:
  virtual ~ConstraintStore () {};
  
  /**
   * Informs the constraint store that something bad happened
   * somewhere else. This forces the store to clean up everything
   * and exit as soon as possible without re-evaluating any constraint.
   */
  virtual void fail () = 0;
  
  /**
   * Sets the satisfiability check during constraint propagation.
   * Thic check increases the time spent for consistency but reduces
   * the total exectuion time.
   * @param sat_check boolean value representing whether or not the
   *        satisfiability check should be performed (default: true).
   */
  virtual void sat_check ( bool sat_check=true ) = 0;
  
  /**
   * It adds the constraints given in input to the queue of
   * constraint to re-evaluate.
   * @param c_id the vector of constraints ids to re-evaluate.
   * @param event the event that has triggered the re-evaluation of 
   *        the given list of constraints.
   * @note only constraints that have been previously attached/imposed
   *       to this constraint store will be re-evaluated.
   */
  virtual void add_changed ( std::vector< size_t >& c_id, EventType event ) = 0;
  
  /**
   * Imposes a constraint to the store. The constraint is added
   * to the list of constraints in this constraint store as well as
   * to the queue of constraint to re-evaluate next call to consistency.
   * Most probably this function is called every time a new constraint
   * is instantiated.
   * @param c the constraint to impose in this constraint store.
   */
  virtual void impose ( ConstraintPtr c ) = 0;
  
  /**
   * Computes the consistency function.
   * This function propagates the constraints that are in the
   * constraint queue until the queue is empty.
   * @return true if all propagate constraints are consistent,
   *         false otherwise.
   */
  virtual bool consistency () = 0;
  
  /**
   * Returns a constraint that is scheduled for re-evaluation.
   * The basic implementation is first-in-first-out. 
   * The constraint is hence remove from the constraint queue, 
   * since it is assumed that it will be re-evaluated right away.
   * @return a const pointer to a constraint to re-evaluate.
   */
  virtual Constraint * getConstraint () = 0;
  
  /**
   * Clears the queue of constraints to re-evaluate.
   * It can be used when implementing different scheme 
   * of constraint propagation.
   */
  virtual void clear_queue () = 0;
  
  /**
   * Returns the total number of constraints in 
   * this constraint store.
   */
  virtual size_t num_constraints () const = 0;
  
  /**
   * Returns the number of constraints to re-evaluate.
   * @return number of constraints to re-evaluate.
   */
  virtual size_t num_constraints_to_reevaluate () const = 0;
  
  /**
   * Returns the total number of propagations performed
   * by this constraint store so far.
   */
  virtual size_t num_propagations () const = 0;
  
  //! Print information about this constraint store.
  virtual void print () const = 0;
};



#endif
