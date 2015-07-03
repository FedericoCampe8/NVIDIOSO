//
//  cuda_model_generator.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "cuda_model_generator.h"
#include "cuda_variable.h"
#include "token_var.h"
#include "token_con.h"
#include "token_sol.h"
#include "fzn_constraint_generator.h"
#include "fzn_search_generator.h"
#include "cuda_simple_constraint_store.h"

using namespace std;

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
	  
    if ( tkn_ptr->get_type() != TokenType::FD_CONSTRAINT ) 
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
    return engine;
}//get_search_engine

ConstraintStorePtr
CudaGenerator::get_store ()
{
    ConstraintStorePtr store;
  
#if CUDAON
    store = make_shared<CudaSimpleConstraintStore> ();
#else
    store = make_shared<SimpleConstraintStore> ();
#endif
  
  return store;
}//get_store



