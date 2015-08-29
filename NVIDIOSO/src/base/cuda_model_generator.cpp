//
//  cuda_model_generator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/09/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "cuda_model_generator.h"
#include "cuda_variable.h"
#include "token_var.h"
#include "token_arr.h"
#include "token_con.h"
#include "token_sol.h"
#include "token_cstore.h"
#include "search_generator.h"
#include "factory_cstore.h"
#include "base_constraint_register.h"
#include "global_constraint_register.h"


using namespace std;

BaseConstraintRegister&   bse_constraint_register = BaseConstraintRegister::get_instance ();
GlobalConstraintRegister& glb_constraint_register = GlobalConstraintRegister::get_instance ();

CudaGenerator::CudaGenerator () :
    _dbg( "CudaGenerator - " ) {
	_obj_var = nullptr;
	LogMsg << _dbg + "Instantiate a model generator" << endl;
}//CudaGenerator

CudaGenerator::~CudaGenerator () {
    _var_lookup_table.clear();
}//~CudaGenerator

std::pair < std::string, std::vector< int > > 
CudaGenerator::get_auxiliary_parameters ( UTokenPtr tkn_ptr )
{
	// Assert current token not nullptr
    assert( tkn_ptr != nullptr );
    
    // Check consistency of current token
    if ( tkn_ptr->get_type() != TokenType::FD_VAR_INFO_ARRAY )
    {
        throw  NvdException ( (_dbg + "Error while instantiating a auxiliary info array").c_str(),  
                              __FILE__, __LINE__ );
    }
    
    /*
     * Get the pointer to the TokenVar.
     * @Todo Change implementation avoiding casting and
     *       decoupling model_generator from tokens.
     * @note Try with Visitor.
     */
    TokenArr * ptr = static_cast<TokenArr *> ( tkn_ptr.get() );
    
    std::vector < std::string > aux_elements = ptr->get_support_elements();
    std::vector < int > aux_int_elements;
    
    // Num rows
    aux_int_elements.push_back ( 1 ); 
    
    // Num columns
    aux_int_elements.push_back ( aux_elements.size () ); 
    
    for ( auto& s : aux_elements )
    {
    	aux_int_elements.push_back ( atoi ( s.c_str() ) ); 
    }
    
    // Set this ID as ID for aux array
    _arr_lookup_table.insert ( ptr->get_var_id () );
    
    return make_pair ( ptr->get_var_id (), aux_int_elements );
}//get_auxiliary_parameters

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
	
	// Store obj var
	if ( ptr->is_objective_var () )
    {
		_obj_var = var_ptr;
	}
	
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
	vector<string> 		par_ptr;
	for ( auto& expr : expr_var_vec )
    {
        auto ptr_aux = _var_lookup_table.find ( expr );
        if ( ptr_aux != _var_lookup_table.end() )
        {
            var_ptr.push_back ( ptr_aux->second );
        }
        auto ptr_arr = _arr_lookup_table.find ( expr );
        if ( ptr_arr != _arr_lookup_table.end() )
        {
        	par_ptr.push_back ( expr );
        }
	}
    
    bool is_digit;
    std::vector<int> var_position_mapping;
    for ( auto& expr : ptr->get_expr_array() )
    {
    	is_digit = true;
    	for ( auto& c : expr )
    	{
    		if ( c != '-' && !(c >= '0' && c <= '9') )
    		{
    			is_digit = false;
    			break;	
    		}
    	}
    	if ( is_digit )
    		var_position_mapping.push_back ( 0 );
    	else
    		var_position_mapping.push_back ( 1 );
    }
    
    for ( auto& expr : ptr->get_expr_not_var_elements_array() )
    {
    	par_ptr.push_back ( expr );
    }

    /*
     * Constraint (pointer) to return.
     * It is initialized with the parameters of the token, i.e., the
     * name, list of pointer to FD variables, and auxiliary arguments.
     */
	if ( tkn_ptr->get_type() == TokenType::FD_CONSTRAINT )
	{
		ConstraintPtr bse_constraint = bse_constraint_register.get_base_constraint ( constraint_name );
		static_cast<BaseConstraint*> ((bse_constraint.get()))->setup ( var_ptr, par_ptr );
		if ( ptr->is_soft () )
    	{
    		bse_constraint->increase_weight ();
    	}
    	
    	// Set position of variables in the constraint as defined from input model
    	bse_constraint->set_var2subscript_mapping ( var_position_mapping );
    	
    	return bse_constraint;
	}
	else
	{// Global constraint
		GlobalConstraintPtr glb_constraint = glb_constraint_register.get_global_constraint ( constraint_name );
		glb_constraint->setup ( var_ptr, par_ptr );
		
		if ( solver_params != nullptr )
		{
			glb_constraint->set_consistency_level ( solver_params->constraint_get_propagator_class () );
		}
		if ( ptr->is_soft () )
    	{
    		glb_constraint->increase_weight ();
    	}
    	
    	// Set position of variables in the constraint as defined from input model
    	glb_constraint->set_var2subscript_mapping ( var_position_mapping );
    	
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
    
    TokenSol * ptr = static_cast<TokenSol *> ( tkn_ptr.get() );
    
    //Sanity check
    assert ( ptr != nullptr );
    
    // Variables to label
    Variable * obj_variable = nullptr;
    vector< Variable * > variables;
     
    // Check whether there is (are) a specific variable(s) to label
    std::string var_goal = ptr->get_var_goal ();
    std::string label_choice = ptr->get_label_choice ();
    std::vector< std::string > vars_to_label = ptr->get_var_to_label();
    if ( label_choice != "" )
    {
    	for ( auto var : _var_lookup_table )
    	{
    		std::string var_in_array{};
    		std::size_t found = var.first.find ( "[" );
    		if ( found != std::string::npos ) 
    		{
    			var_in_array = var.first.substr ( 0, found );
    		}
    		 
    		if ( var.first == label_choice || var_in_array == label_choice )
    		{
        		variables.push_back ( (var.second).get() );
        	}
    	}
    }
    else if ( vars_to_label.size() != 0 )
    {
    	for ( auto var : vars_to_label )
    	{
    		auto it = _var_lookup_table.find ( var );
    		if ( it != _var_lookup_table.end () )
    		{
    			variables.push_back ( (it->second).get() );
    		}
    	}
    }
    else
    {
    	for ( auto var : _var_lookup_table )
    	{	
    		// Do not label objective variable
    		if ( _obj_var != nullptr && var.first == _obj_var->get_str_id () )
    		{
    			continue;
    		}
        	variables.push_back ( (var.second).get() );
    	}
    }
    
    if ( var_goal != "" )
    {
    	auto it = _var_lookup_table.find ( var_goal );
    	if ( it != _var_lookup_table.end () )
    	{
    		obj_variable = (it->second).get();
    	}
    	else
    	{
    		LogMsg.error( _dbg + "Objective variable not found",
                      	  __FILE__, __LINE__);
        	return nullptr;
    	}
    }
  
    struct  SortingFunction
    {
        bool operator() ( Variable * a, Variable * b ) {
            return a->get_id() < b->get_id();
        }
    } MySortingFunction;
    std::sort( variables.begin(), variables.end(), MySortingFunction );
  
    // Get search engine according to the input model
    SearchEnginePtr engine = SearchFactory::get_search_shr_ptr ( variables, ptr, obj_variable );
    variables.clear ();

    // Set solver parameters
    if ( solver_params != nullptr )
    {
        engine->set_debug               ( solver_params->search_get_debug() );
        engine->set_trail_debug         ( solver_params->search_get_trail_debug () );
        engine->set_time_watcher        ( solver_params->search_get_time_watcher () );
        engine->set_timeout_limit       ( solver_params->search_get_timeout () );
        engine->set_solution_limit      ( solver_params->search_get_solution_limit () );
        engine->set_backtrack_out       ( solver_params->search_get_backtrack_limit () );
        engine->set_nodes_out           ( solver_params->search_get_nodes_limit () );
        engine->set_wrong_decisions_out ( solver_params->search_get_wrong_decision_limit () );
    }
    
    return engine;
}//get_search_engine
 
ConstraintStorePtr
CudaGenerator::get_store ( UTokenPtr tkn_ptr )
{
    ConstraintStorePtr store;
  	
  	int  cstore_type  = 0;
  	bool on_device    = false;
  	bool local_search = false;
  	
  	if ( tkn_ptr != nullptr )
  	{
  		TokenCStore * ptr = static_cast<TokenCStore *> ( tkn_ptr.get() );
    
    	//Sanity check
    	assert ( ptr != nullptr );
    	local_search = ptr->on_local_search ();
  	}
  	
#if CUDAON

	on_device = true;
	if ( solver_params != nullptr )
    {
    	cstore_type = solver_params->cstore_type_to_int ( solver_params->cstore_get_dev_propagation () );
	}
	
#endif

	
	store = FactoryCStore::get_cstore ( on_device, cstore_type, local_search );
	
    // Set solver parameters
    if ( solver_params != nullptr )
    {
        store->sat_check ( solver_params->cstore_get_satisfiability () );
        store->con_check ( solver_params->cstore_get_consistency () );
        
        if ( local_search )
        {	
        	// Set constraints sat type (default: mixed)
        	if ( solver_params->cstore_constraints_all_soft () )
        	{
        		(std::static_pointer_cast<SoftConstraintStore>( store ))->impose_all_soft ();
        	}
        	else if ( solver_params->cstore_constraints_all_hard () )
        	{
        		(std::static_pointer_cast<SoftConstraintStore>( store ))->impose_all_hard ();
        	}
        }
        
#if CUDAON

		(std::static_pointer_cast<CudaSimpleConstraintStore>( store ))->
		set_max_block_size ( solver_params->cuda_get_max_block_size () );
		
		(std::static_pointer_cast <CudaSimpleConstraintStore> ( store ))->
		set_prop_loop_out ( solver_params->cstore_get_dev_loop_out () );
		
#endif

    }
    
    return store;
}//get_store



