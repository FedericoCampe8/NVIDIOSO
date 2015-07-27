//
//  cuda_simple_constraint_store_1b1v.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/21/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using 1 block per variable.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_1b1v__
#define __NVIDIOSO__cuda_simple_constraint_store_1b1v__

#include "cuda_simple_constraint_store.h"

class CudaSimpleConstraintStore1b1v : public CudaSimpleConstraintStore {
private:
    //! Size of the array of constraints that will be coipied on device
    std::size_t _h_constraint_queue_size;

    //! Current size of constraint queue to allocate on device
    std::size_t _h_con_idx_queue_size;

    //! Array of indeces for the constraint queue
    std::vector<int> _h_constraint_queue_idx;

	//! Auxiliary array of states
	uint * _d_states_aux;
	
    //! Array of indeces for constraint queue on device
    int * _d_constraint_queue_idx;
    
    /**
     * Mapping between variables (ids) and (set of) constraints
     * in which the variables is involved.
     * For each constraint it is also stored the index (relative position)
     * of that variable in the scope of the constraint.
     * Therefore, each var x will have a vector of size 2n where n is the
     * number of constraints where x is involved into:
     * V_x -> | C_i | pos in C_i | C_{i+1} | pos in C_{i+1} | ... 
     */
    std::unordered_map < int, std::vector<int> > _var_to_constraint;

    //! Pointers (indeces) to the list of constraints for each variable
    std::vector<int> _var_to_constraint_idx;

    //! Reset all mappings
    void reset ();
    
    /**
     * Sequential propagation to perform if parallel propagation
     * cannot be performed for some external reasons.
     */
    void sequential_propagation ();
    
protected:
    //! Add the constraint to the queue of constraints to propagate for each variable
    void add_changed ( size_t c_id, EventType event ) override;
    
    //! Override to copy auxiliary states also 
	bool move_states_to_device () override;
    
    //! Override to copy back auxiliary state also 
    bool move_states_from_device () override;
    
    //! Copy the current queue of constraints on device
    void move_queue_to_device () override;
    
    //! Invoke the kernel which performs consistency on device
    void dev_consistency ();

public:
    CudaSimpleConstraintStore1b1v ();

    ~CudaSimpleConstraintStore1b1v ();
	
	using CudaSimpleConstraintStore::add_changed;
	
    void finalize ( CudaCPModel* ptr ) override;
};

#endif
