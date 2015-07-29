//
//  cuda_simple_constraint_store_1b1c.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  This class implements a (simple) constraint store on device 
//  using 1 block per constraint.
//

#ifndef __NVIDIOSO__cuda_simple_constraint_store_1b1c__
#define __NVIDIOSO__cuda_simple_constraint_store_1b1c__

#include "cuda_simple_constraint_store.h"

class CudaSimpleConstraintStore1b1c : public CudaSimpleConstraintStore {
private:
    /**
     * Flag forcing the upated of domains on all the variables.
     * @note no computation to determine the changed variables is performed
     *       when this flag is set to True.
     *       However, updating of variables is peformed on all domains.
     */
    bool _update_all;
    
    //! Number of loops of propagation performed 
    int _update_var_iter;

    //!  Pointers to domain's states
    uint * _domains_ptr;

    // Array of indeces to states to be updated 
    int * _domain_idx;
    
    //! Pointers to indeces on domains in _domains_ptr
    int * _domains_idx_ptr;
    
    //! Set of variable ids to update after propagation on device
    std::unordered_set<int> _updating_vars_set;

    /**
     * Set the set of variables to updated w.r.t. their reduced domains
     * after propagation on device.
     * @note Use this function to updated only a subset of all the variables.
     *       This variables are the variables which will be considered on the next
     *       propagation's loop or when copied back to the host (from device).
     */
    void set_vars_to_update_after_prop ();

    /**
     * Sequential propagation to perform if parallel propagation
     * cannot be performed for some external reasons.
     */
    void sequential_propagation ();
    
protected:
    //! Invoke the kernel which performs consistency on device
    void dev_consistency () override;

    void init_store () override;
    
public:
    CudaSimpleConstraintStore1b1c ();

    ~CudaSimpleConstraintStore1b1c ();
    
    void finalize ( CudaCPModel* ptr ) override;
};

#endif
