//
//  simple_constraint_store.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store.
//

#ifndef __NVIDIOSO__simple_constraint_store__
#define __NVIDIOSO__simple_constraint_store__

#include "constraint_store.h"

class SimpleConstraintStore : public ConstraintStore {
protected:
    //! Debug info
    std::string _dbg;
  
    /**
     * Mapping between constraints' ids and
     * constraints' pointer. Any new constraint
     * imposed into the store is stored here.
     */
    std::unordered_map < size_t, ConstraintPtr > _lookup_table;
  
    /**
     * Stores the constraints for which reevaluation is needed.
     * It represents the constraint_queue. It does not register
     * constraints that are already in the constraint queue.
     * @note there is only a queue in this simple constraint store.
     *       Other implementations may consider to use multiple
     *       constraint queue (e.g., one for each domains'event).
     */
    std::set< size_t > _constraint_queue;
  
    //! Current constraint to reevaluate.
    Constraint* _constraint_to_reevaluate;
  
    //! Number of constraints in the _constraint_queue.
    size_t _constraint_queue_size;
  
    //! Number of constraints imposed into the store.
    size_t _number_of_constraints;
  
    //! Number of propagations performed so far.
    size_t _number_of_propagations;
  
    /** 
     * Defines whether the satisfiability check 
     * should be performed or not (default: true).
     */
    bool _satisfiability_check;

    /**
     * Defines whether the consistency propagation
     * should be performed or not (default: true).
     */
    bool _consistecy_propagation;
    
    /**
     * Keeps track whether some failure happened during
     * some operations on this constraint store.
     */
    bool _failure;
  
    //! Handle a failure state
    virtual void handle_failure ();
  
    //! Add a single constraint for re-evaluation.
    virtual void add_changed ( size_t c_id, EventType event );
  
public:
    /**
     * Default constructor. It initializes the 
     * internal data structures of this constraint store.
     */
    SimpleConstraintStore ();
  
    virtual ~SimpleConstraintStore ();
  
    /**
     * Informs the constraint store that something bad happened
     * somewhere else. This forces the store to clean up everything
     * and exit as soon as possible without re-evaluating any constraint.
     */
    void fail () override;
  
    /**
     * Sets the satisfiability check during constraint propagation.
     * Thic check increases the time spent for consistency but reduces
     * the total exectuion time.
     * @param sat_check boolean value representing whether or not the
     *        satisfiability check should be performed (default: true).
     */
    void sat_check ( bool sat_check=true ) override;

    /**
     * Sets consistecy propagation to con_check.
     * @param con_check boolean value. True if constraint propagation
     *        has to be performed, False otherwise.
     */
    void con_check ( bool con_check=true ) override;
    
    /**
     * It adds the constraints given in input to the queue of
     * constraint to re-evaluate.
     * @param c_id the vector of constraints ids to re-evaluate.
     * @param event the event that has triggered the re-evaluation of
     *        the given list of constraints.
     * @note only constraints that have been previously attached/imposed
     *       to this constraint store will be re-evaluated.
     */
    void add_changed ( std::vector< size_t >& c_id, EventType event ) override;
  
    /**
     * Imposes a constraint to the store. The constraint is added
     * to the list of constraints in this constraint store as well as
     * to the queue of constraint to re-evaluate next call to consistency.
     * Most probably this function is called every time a new constraint
     * is instantiated.
     * @param c the constraint to impose in this constraint store.
     * @note if c is already in the list of constraints in this 
     *       constraint store, it won't be added again nor re-evaluated.
     */
    void impose ( ConstraintPtr c ) override;
  
    /**
     * Computes the consistency function.
     * This function propagates the constraints that are in the
     * constraint queue until the queue is empty.
     * @return true if all propagate constraints are consistent,
     *         false otherwise.
     */
    bool consistency () override;
  
    /**
     * Returns a constraint that is scheduled for re-evaluation.
     * The basic implementation is first-in-first-out.
     * The constraint is hence remove from the constraint queue,
     * since it is assumed that it will be re-evaluated right away.
     * @return a const pointer to a constraint to re-evaluate.
     */
    Constraint * getConstraint () override;
  
    /**
     * Clears the queue of constraints to re-evaluate.
     * It can be used when implementing different scheme
     * of constraint propagation.
     */
    void clear_queue () override;
  
    /**
     * Returns the total number of constraints in
     * this constraint store.
     */
    size_t num_constraints () const override;
  
    /**
     * Returns the number of constraints to re-evaluate.
     * @return number of constraints to re-evaluate.
     */
    size_t num_constraints_to_reevaluate () const override;
  
    /**
     * Returns the total number of propagations performed
     * by this constraint store so far.
     */
    size_t num_propagations () const override;
  
    //! Print infoformation about this simple constraint store.
    void print () const override;
};

#endif /* defined(__NVIDIOSO__simple_constraint_store__) */
