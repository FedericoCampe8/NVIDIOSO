
//
//  cuda_simple_constraint_store_1b1c.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/18/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "cuda_simple_constraint_store_1b1c.h"
#include "cuda_propagation_utilities.h"
#include "cuda_synchronization_utilities.h"

using namespace std;

CudaSimpleConstraintStore1b1c::CudaSimpleConstraintStore1b1c () :
    CudaSimpleConstraintStore (),
    _update_all               ( false ),
    _update_var_iter          ( 0 ) {
    _dbg = "CudaSimpleConstraintStore1b1c - ";
}//CudaSimpleConstraintStore1b1c

CudaSimpleConstraintStore1b1c::~CudaSimpleConstraintStore1b1c () {

#if CUDAON

    logger.cuda_handle_error ( cudaFree( _domain_idx ) );
    
#endif

}//~CudaSimpleConstraintStore1b1c

void
CudaSimpleConstraintStore1b1c::finalize ( CudaCPModel* ptr )
{
    CudaSimpleConstraintStore::finalize ( ptr );
    if ( _cp_model_ptr == nullptr )
    {
    	std::string err = _dbg + "finalize: CudaCPModel NULL.\n";
    	LogMsg << err;
        throw NvdException ( err.c_str(), __FILE__, __LINE__ );
    }
    
    _domains_ptr     = _cp_model_ptr->get_dev_domain_states_ptr ();
    _domains_idx_ptr = _cp_model_ptr->get_dev_domain_index_ptr ();
    
#if CUDAON

    dev_grid_size.x  = _cp_model_ptr->num_variables ();
	
    if ( logger.cuda_handle_error ( cudaMalloc( (void**)&_domain_idx, _cp_model_ptr->num_variables () * sizeof ( int ) ) ) )
    {
        _domain_idx = nullptr;
    }
    
#endif
}//finalize

void
CudaSimpleConstraintStore1b1c::init_store ()
{
    if ( _update_all )
    {
        _update_var_iter = _loop_out;
    }
    else
    {
        _update_var_iter = 0;
    }
}//init_store

void
CudaSimpleConstraintStore1b1c::set_vars_to_update_after_prop ()
{

#if CUDAON
    
    /*
     * Implementation of updating vars w.r.t. to the
     * number of propagatsion's loop as follow:
     * 1    - Only the vars in the scope of the propagated constraints
     * 2    - Only the vars in 1 and the vars involved in constraints
     *        related to the vars in 1
     * >= 3 - All variables
     * @todo Point 2.
     */
    if ( _update_var_iter == 0 )
    { 
        _updating_vars_set.clear();
        
        std::unordered_set<int> vars;
        for ( auto& c_id : _constraint_queue )
        {
            auto c_scope = _lookup_table[ c_id ]->scope ();
            for ( auto& var : c_scope )
            {
                _updating_vars_set.insert ( var->get_id () );
            }
        }
        vector <int> map_vars_to_doms = _cp_model_ptr->dev_var_mapping ( _updating_vars_set );
        if ( logger.cuda_handle_error ( cudaMemcpy (_domain_idx, &map_vars_to_doms[ 0 ],
                                                    _updating_vars_set.size() * sizeof (int),
                                                    cudaMemcpyHostToDevice ) ) )
        {
            _domains_idx_ptr = _cp_model_ptr->get_dev_domain_index_ptr ();
            dev_grid_size.x  = _cp_model_ptr->num_variables ();
        }
        else
        {
            _domains_idx_ptr = _domain_idx;
            dev_grid_size.x  = _updating_vars_set.size();
        }
        
    }
    else
    {// All variables to update
        dev_grid_size.x  = _cp_model_ptr->num_variables ();
        _domains_ptr     = _cp_model_ptr->get_dev_domain_states_ptr ();
        _domains_idx_ptr = _cp_model_ptr->get_dev_domain_index_ptr ();
    }
    _update_var_iter++;
    
#endif

}//set_vars_to_update_after_prop

void
CudaSimpleConstraintStore1b1c::sequential_propagation ()
{

#if CUDAON
    
    static bool advice = true;
    if ( advice )
    {
        std::cout << _dbg + "Shared memory limit on device reached:\n";
        std::cout << "The current version of iNVIDIOSO uses sequential propagation\n";
        std::cout << "on device when the limit of " << _shared_limit << " bytes\n";
        std::cout << "for shared memory is reached.\n";
        std::cout << "Press any key to continue (this message won't be dysplayed again)\n";
        getchar();
        advice = false;
    }
    CudaPropUtils::cuda_consistency_sequential <<< 1, 1, 0 >>> 
    ( _d_constraint_queue, _constraint_queue.size() );
    
#endif
    
}//sequential_propagation

void
CudaSimpleConstraintStore1b1c::dev_consistency ()
{
	
#if CUDAON

    if ( _scope_state_size > _shared_limit )
    {
        sequential_propagation ();
    }
    else
    {
        dev_grid_size.x  = _constraint_queue.size();
        CudaPropUtils::cuda_consistency_1b1c <<< dev_grid_size, dev_block_size, _scope_state_size >>>
        ( _d_constraint_queue );

        // Calculate the set of variables modified by the previous propagation
        set_vars_to_update_after_prop ();
	
        /*
         * Synchronize domains and make them consistent
         * @todo synchronize only the variable that have effectively changed
         */
        CudaSynchUtils::cuda_set_domains_from_bit_1b1v <<< dev_grid_size, dev_block_size, STANDARD_DOM >>>
        ( _domains_idx_ptr, _domains_ptr );
    }
    
#endif

}//dev_consistency



