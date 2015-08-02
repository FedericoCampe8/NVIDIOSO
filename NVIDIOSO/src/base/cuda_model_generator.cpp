//
//  cuda_model_generator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_model_generator.h"
#include "cuda_variable.h"
#include "token_var.h"
#include "token_con.h"
#include "token_sol.h"
#include "fzn_constraint_generator.h"
#include "fzn_search_generator.h"
#include "factory_cstore.h"

using namespace std;

GlobalConstraintRegister& glb_constraint_register = GlobalConstraintRegister::get_instance ();

CudaGenerator::CudaGenerator () :
    _dbg( "CudaGenerator - " )
{
	LogMsg << _dbg + "Instantiate a model generator" << endl;
}//CudaGenerator

CudaGenerator::~CudaGenerator () {
    _var_lookup_table.clear();
}//~CudaGenerator

VariablePtr
CudaGenerator::get_variable ( UTokenPtr tkn_ptr )
{  
    // Assert current token not nullptr
    assert( tkn_ptr != nullptr );
    
    // Check consistency of current token
    if ( (tkn_ptr->get_type() != TokenType::FD_VARIABLE) &&
         (tkn_ptr->get_type() != TokenType::FD_VAR_ARRAY) )
    {
        throw  NvdException ( (_dbg + "Error while instantiating a FD Variables").c_str(),  
                              __FILE__, __LINE__ );
    }

    /*
     * Get the pointer to the TokenVar.
     * @Todo Change implementation avoiding casting and
     *       decoupling model_generator from tokens.
     * @note Try with Visitor.
     */
    TokenVar * ptr = static_cast<TokenVar *> ( tkn_ptr.get() );
    
    /*
     * Variable (pointer) to return.
     * It is initialized with the int id of the token, i.e., the 
     * unique identifier given when the string was parsed from the model.
     */
    VariablePtr var_ptr;
    try
    {
        var_ptr = make_shared<CudaVariable> ( tkn_ptr->get_id() );
    }
    catch ( NvdException& e )
    {
        cout << e.what() << endl;
        throw;
    }
  
    // Set string id, i.e., the string id as reported in the model
    var_ptr->set_str_id( ptr->get_var_id () );
  
    // Set type of the variable
    if ( ptr->is_objective_var () )
    {
        var_ptr->set_type( VariableType::OBJ_VARIABLE );
    }
    else if ( ptr->is_support_var () )
    {
        var_ptr->set_type( VariableType::SUP_VARIABLE );
    }
    else
    {
        var_ptr->set_type( VariableType::FD_VARIABLE );
    }
  
    /*
     * Set domain of the variable.
     * @note only int (range) domains on CUDA.
     * @note tkn_ptr is not given as input to the function "set_domain".
     *       This is done in order to decouple tokens from domains.
     */
    var_ptr->set_domain_type ( DomainType::INTEGER );
  
    /*
     * Switch in order to set the actual domain for the current variable,
     * depending on the (variable) token type.
     */
    switch ( ptr->get_var_dom_type() )
    {
        case VarDomainType::BOOLEAN:
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain( 0, 1 );
            break;
        case VarDomainType::INTEGER: 
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain ();
            break;
        case VarDomainType::FLOAT:
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->set_domain ();
            break;
        case VarDomainType::RANGE: 
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->
                set_domain ( ptr->get_lw_bound_domain(),
                             ptr->get_up_bound_domain() );
            break;
        case VarDomainType::SET: 
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->
                set_domain ( ptr-> get_subset_domain() );
            break;
        case VarDomainType::SET_INT:
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->
                set_domain ( ptr->get_subset_domain() );
            break;
        case VarDomainType::SET_RANGE:
            (std::static_pointer_cast<CudaVariable>( var_ptr ))->
                set_domain ( ptr->get_subset_domain() );
            break; 
        default:
            /*
             * Return a nullptr to check in order to further
             * specilize this method if other types will be added.
             */
            return nullptr;
    }
  
    // Store the string id of the current variable in the lookup table.
    _var_lookup_table [ var_ptr->get_str_id() ] = var_ptr;
  
    return var_ptr;
}//get_variable

ConstraintPtr
CudaGenerator::get_constraint ( UTokenPtr tkn_ptr )
{ 
    // Error handling
	assert( tkn_ptr != nullptr );
	  
    if ( tkn_ptr->get_type() != TokenType::FD_CONSTRAINT  && 
         tkn_ptr->get_type() != TokenType::FD_GLB_CONSTRAINT ) 
    {
	   	LogMsg.error( _dbg + "Error while instantiating a Constraint",
        	          __FILE__, __LINE__);
    	return nullptr;
    }

	/*
 	 * Get the pointer to the tokenCon.
 	 * @Todo Change implementation avoiding casting and
   	 *       decoupling model_generator from tokens.
  	 * @note Try with Visitor.
 	 */
   	TokenCon * ptr = static_cast<TokenCon *> ( tkn_ptr.get() );
    
  	string constraint_name = ptr->get_con_id ();
  
  	vector<string> expr_var_vec = ptr->get_expr_var_elements_array();
  
  	/*
  	 * Check which string is the id of a variable and
 	 * set the list of var pointer accordingly.
 	 */
	vector<VariablePtr> var_ptr;
	for ( auto expr : expr_var_vec )
    {
        auto ptr_aux = _var_lookup_table.find ( expr );
        if ( ptr_aux != _var_lookup_table.end() )
        {
            var_ptr.push_back ( ptr_aux->second );
        }
   	 }
    
    vector<string> params_vec = ptr->get_expr_not_var_elements_array();
  
    /*
     * Constraint (pointer) to return.
     * It is initialized with the parameters of the token, i.e., the
     * name, list of pointer to FD variables, and auxiliary arguments.
     */
	if ( tkn_ptr->get_type() == TokenType::FD_CONSTRAINT )
	{
		try
    	{
        	return
            FZNConstraintFactory::get_fzn_constraint_shr_ptr( constraint_name,
                                                              var_ptr,
                                                              params_vec );
    	}
    	catch ( NvdException& e )
    	{
        	cout << e.what() << endl;
        	throw;
    	}
	}
	else
	{
		GlobalConstraintPtr glb_constraint = glb_constraint_register.get_global_constraint ( constraint_name );
		glb_constraint->setup ( var_ptr, params_vec );
		if ( solver_params != nullptr )
		{
			glb_constraint->set_propagator_class ( solver_params->constraint_get_propagator_class () );
		}
		return glb_constraint;
	}
}//get_constraint

SearchEnginePtr
CudaGenerator::get_search_engine ( UTokenPtr tkn_ptr )
{  
    // Error handling
    assert( tkn_ptr != nullptr );
  
    if ( tkn_ptr->get_type() != TokenType::FD_SOLVE )
    {
        LogMsg.error( _dbg + "Error while instantiating a Search Engine",
                      __FILE__, __LINE__);
        return nullptr;
    }
    
    // Variables to label
    vector< Variable * > variables;
    for ( auto var : _var_lookup_table )
    {
        variables.push_back ( (var.second).get() );
    }
  
    struct  SortingFunction
    {
        bool operator() ( Variable * a, Variable * b ) {
            return a->get_id() < b->get_id();
        }
    } MySortingFunction;
    std::sort( variables.begin(), variables.end(), MySortingFunction );
  
    // Get search engine according to the input model
    SearchEnginePtr engine =
        FZNSearchFactory::get_fzn_search_shr_ptr ( variables,
                                                   static_cast<TokenSol * >(tkn_ptr.get()) );
    variables.clear ();

    // Set solver parameters
    if ( solver_params != nullptr )
    {
        engine->set_debug          ( solver_params->search_get_debug() );
        engine->set_trail_debug    ( solver_params->search_get_trail_debug () );
        engine->set_time_watcher   ( solver_params->search_get_time_watcher () );
        engine->set_timeout_limit  ( solver_params->search_get_timeout () );
        engine->set_solution_limit ( solver_params->search_get_solution_limit () );
        engine->set_backtrack_out  ( solver_params->search_get_backtrack_limit () );
        engine->set_nodes_out      ( solver_params->search_get_nodes_limit () );
        engine->set_wrong_decisions_out ( solver_params->search_get_wrong_decision_limit () );
    }
    
    return engine;
}//get_search_engine

ConstraintStorePtr
CudaGenerator::get_store ()
{
    ConstraintStorePtr store;
  
#if CUDAON
	if ( solver_params != nullptr )
    {
		store = FactoryCStore::get_cstore ( true, 
	 					 					solver_params->cstore_type_to_int (
						 					solver_params->cstore_get_dev_propagation () ) );
	}
	else
	{// Default Constraint Store on device
		store = FactoryCStore::get_cstore ( true );
	}
    
#else

	store = FactoryCStore::get_cstore ();
	
#endif

    // Set solver parameters
    if ( solver_params != nullptr )
    {
        store->sat_check ( solver_params->cstore_get_satisfiability () );
        store->con_check ( solver_params->cstore_get_consistency () );
        
#if CUDAON

		(std::static_pointer_cast<CudaSimpleConstraintStore>( store ))->
		set_max_block_size ( solver_params->cuda_get_max_block_size () );
		
		(std::static_pointer_cast <CudaSimpleConstraintStore> ( store ))->
		set_prop_loop_out ( solver_params->cstore_get_dev_loop_out () );
		
#endif

    }
    
    return store;
}//get_store



