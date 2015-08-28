//
//  soft_constraint_store.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store which handles both
//  hard and soft constraint. 
//  It always perform propagation on ALL constraints in the constraint queue.
//

#ifndef __NVIDIOSO__soft_constraint_store__
#define __NVIDIOSO__soft_constraint_store__

#include <boost/dynamic_bitset.hpp>
#include "simple_constraint_store.h"

class SoftConstraintStore;
typedef std::unique_ptr<SoftConstraintStore> SoftConstraintStoreUPtr; 
typedef std::shared_ptr<SoftConstraintStore> SoftConstraintStoreSPtr;  

class SoftConstraintStore : public SimpleConstraintStore {
protected:
  
  	/**
  	 * Set of all soft constraints ids.
  	 * This set is filled when a new constraint is added to
  	 * the constraint store and it is used to determine whether
  	 */
	std::unordered_set < std::size_t > _soft_constraint_set;
	
	/**
  	 * Set of all hard constraints ids.
  	 * This set is filled when a new constraint is added to
  	 * the constraint store and it is used to determine whether
  	 */
	std::unordered_set < std::size_t > _hard_constraint_set;
	
	/**
	 * Lookup table mapping constraint ids into positions in a bitset
	 * array which is used to determine the number of unsatisfied constraints 
	 * at each propagation step.
	 */
	std::unordered_map < std::size_t, std::size_t > _constraint_2_bitset;
	
	/**
	 * Hash table mapping satisfied/non-satisfied constraints to their 
	 * level of unsatisfiability.
	 * This hash table is used at each invokation of the consistency function to 
	 * determine the level of unsatisfied constraints represented as a double.
	 */
	std::unordered_map < std::size_t, double > _sat_constraint_level; 
	
	/**
	 * Bitset mapping satisfied/non-satisfied constraints.
	 * Bitset is used at each invokation of the consistency function to 
	 * determine the number of unsatisfied constraints.
	 */
	boost::dynamic_bitset<> _sat_constraint_bitset; 
	
	//! Force all constraints to be considered as soft constraints
	bool _all_soft_constraints;
	
	//! Force all constraints to be considered as hard constraints
	bool _all_hard_constraints;
	
	//! Soft consistency imposed
	bool _force_soft_consistency;
	
	//! Degree of constraint unsatisfiability at each consistency invokation
	mutable double _unsat_constraint_level;
	
	//! Number of unsatisfied hard constraint at each consistency invokation
	mutable std::size_t _unsat_constraint_counter;
	
	std::size_t _bitset_index;
	
	//! Reset failure parameter as not failed
	void reset_failure ();
	
	/**
	 * Returns true if c is a hard constraint.
	 * @param c pointer to a constraint
	 * @return true if c is considered as hard constraint.
	 */
	bool is_hard ( Constraint* c ) const;
  	  
	/**
	 * Returns true if c is a soft constraint.
	 * @param c pointer to a constraint
	 * @return true if c is considered as soft constraint.
	 */
	bool is_soft ( Constraint* c ) const;
	 
	/**
	 * Reset counters and initialize the bitset map for satisfied constraints.
	 * Bitset is initialized only the first time consistency is being invoked.
	 */
	virtual void reset_unsat_counters ();
	
	/**
	 * Set counters considering the current values stored in the bitset and 
	 * the total unsat level value.
	 * @param force_update Boolean value that forces the update of the counters
	 *        by reading directly from the hash tables (i.e., bitset and unordered map),
	 *        the values for the counter taking O(n) time where n is the total number
	 *        of attached constraints. By default this values are preserved during 
	 *        computation and this method runs in O(1) time.
	 * @note  forcing the update of the counters may be needed after failure.
	 */
	virtual void set_unsat_counters ( bool force_update=false );
	
	/**
	 * Set as unsatisfied (into the bitset) the bit corresponding to the
	 * constraint given in input.
	 * @param c pointer to the unsatisfied constraint.
	 * @param unsat true is the constraint is unsat (default), false otherwise.
	 */
	virtual void record_unsat_value ( Constraint* c, bool sat=false );
	
	/**
	 * Set as unsatisfied (into the bitset) the bit corresponding to the
	 * constraint given in input.
	 * @param c pointer to the unsatisfied constraint.
	 * @param unsat true is the constraint is unsat (default), false otherwise.
	 */
	virtual void record_unsat_constraint ( Constraint* c, bool sat=false );
	
public:

    /**
     * Default constructor. It initializes the 
     * internal data structures of this constraint store.
     */
    SoftConstraintStore ();
  
    virtual ~SoftConstraintStore ();
    
    //! Initialize internal state of the store (e.g., bitset, hash tables, counters, etc.)
    virtual void initialize_internal_state ();
    
    /**
     * Resets internal state of this constraints store, i.e.,
     * resets bitset and counters.
     * @note this method DOES NOT detach any constraint.
     *       All constraints currently attached to this constraint store remain attached.
     * @note this method empty the queue of constraints.
     * @note this method is supposed to be called before the first constraint propagation.
     */
    void reset_state ();
    
    /**
     * Consider each imposed constraint as a soft constraint.
     * @note default is mixed.
     */
    void impose_all_soft ();
    
    /**
     * Consider each imposed constraint as a hard constraint.
     * @note default is mixed.
     */
    void impose_all_hard ();
    
    /**
     * Forces consistency to be performed as all constraints were soft.
     * @param force_soft Boolean value forcing soft consistency on all constraints.
     */
    void force_soft_consistency ( bool force_soft = true );
     
    /**
     * Imposes a constraint on the store. The constraint is added
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
     * Returns a hard constraint that is scheduled for re-evaluation.
     * The basic implementation is first-in-first-out.
     * The constraint is hence remove from the constraint queue,
     * since it is assumed that it will be re-evaluated right away.
     * @return a const pointer to a constraint to re-evaluate.
     * @note it returns NULL if no hard constraint needs to be re-evaluated.
     */
    Constraint * get_hard_constraint ();
     
    /**
     * Returns a soft constraint that is scheduled for re-evaluation.
     * The basic implementation is first-in-first-out.
     * The constraint is hence remove from the constraint queue,
     * since it is assumed that it will be re-evaluated right away.
     * @return a const pointer to a constraint to re-evaluate.
     * @note it returns NULL if no soft constraint needs to be re-evaluated.
     */
    Constraint * get_soft_constraint ();
  
    /**
     * Returns the total number of soft constraints in
     * this constraint store.
     */
    std::size_t num_soft_constraints () const;
  
  	/**
     * Returns the total number of hard constraints in
     * this constraint store.
     */
    std::size_t num_hard_constraints () const;
    
    /**
     * Returns the total number of soft constraints 
     * which are not satisfied after propagation.
     * @note If a constraint is hard and it is not satisfied,
     *       constraint store returns false;
     */
    std::size_t num_unsat_constraints () const;
    
    /**
     * Returns the index/level calculated on all 
     * hard and soft constraints representing how much they are
     * satisfied by constraint propagation.
     */
    double get_unsat_level_constraints () const;
    
    //! Print infoformation about this simple constraint store.
    void print () const override;
};

#endif /* defined(__NVIDIOSO__simple_constraint_store__) */
