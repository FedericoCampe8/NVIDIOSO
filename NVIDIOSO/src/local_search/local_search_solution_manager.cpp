//
//  local_search_solution_manager.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "local_search_solution_manager.h"

using namespace std;

LocalSearchSolutionManager::LocalSearchSolutionManager () :
	SimpleSolutionManager () {
	_objective_epsilon     = 0.0;
	_objective_val         = std::numeric_limits<double>::max();
	_objective_sat         = true;
	_unsat_constraints_out = 0;
	_objective_var         = nullptr;
}//LocalSearchSolutionManager

LocalSearchSolutionManager::LocalSearchSolutionManager ( std::vector < Variable* >& vars, Variable* obj_var ) :
	LocalSearchSolutionManager() {
	if ( vars.size() )
    	for ( auto var : vars )
      		_variables[ var->get_id() ] = var;
	_objective_var = obj_var;
	
	if ( _objective_var != nullptr )
	{
		_objective_sat = false;
	} 
}//LocalSearchSolutionManager

LocalSearchSolutionManager::~LocalSearchSolutionManager () {
}//~LocalSearchSolutionManager 
 
void
LocalSearchSolutionManager::set_obj_variable ( Variable* obj_var ) 
{
	_objective_var = obj_var;
	
	if ( _objective_var != nullptr )
	{
		_objective_sat = false;
	} 
}//set_obj_variable
 
void 
LocalSearchSolutionManager::set_epsilon_limit ( double epsilon )
{
	_objective_epsilon = epsilon;
}//set_epsilon_limit

void 
LocalSearchSolutionManager::use_satisfiability_obj ( bool sat )
{  
	_objective_sat = sat;
}//use_satisfiability_obj 
  
bool 
LocalSearchSolutionManager::epsilon_satisfied ()
{
	if ( !_objective_var->is_singleton() )
	{
		throw
      	NvdException ( string("epsilon_satisfied: obective variable not singleton.").c_str() );
	}
	
	double obj_upd_val = _objective_var->domain_iterator->min_val();
	if ( abs( obj_upd_val - _objective_val ) < _objective_epsilon )
	{
		_objective_val = obj_upd_val;
		return true;
	}
	_objective_val = obj_upd_val;
	return false;
}//epsilon_satisfied

bool
LocalSearchSolutionManager::constraint_satisfied ( std::size_t sat_con )
{
	if ( sat_con == _unsat_constraints_out )
	{
		return true;
	}
	return false;
}//constraint_satisfied

bool
LocalSearchSolutionManager::notify_on_propagation ( std::size_t value ) 
{
	if ( !_variables.size() ) return true;
      
   	/*
   	 * Satisfiability as objective value and constraints are satisfied or
   	 * Objective variable and epsilon is reached.
   	 */
  	if ( ( _objective_sat && constraint_satisfied ( value )) ||
  	     ( !_objective_sat && epsilon_satisfied () ) ) 
  	{
  		string solution_str = "";
  		for ( auto var : _variables ) 
  		{
    		if ( !var.second->is_singleton() ) 
    		{
      			throw
      			NvdException ( string("Something went wrong: not all variables assigned for solution.").c_str() );
    		}
    		solution_str += var.second->domain_iterator->get_string_representation ();
  		} 
 
  		_solution_strings.push_back ( solution_str );
  	
  		_number_of_solutions++;
  	}
  	
  	if ( _find_all_solutions ) return false;
  	
  	// Max number of solutions reached
  	if ( _number_of_solutions >= _max_number_of_solutions ) return true;
  
  	if ( _find_all_solutions ) return false;
  
  	return false;
}//notify_on_propagation

void
LocalSearchSolutionManager::print () const {
	cout << "Local Search Solution Manager:\n";
	cout << "Use satisfiability as objective:  ";
	if ( _objective_sat ) cout << "YES\n";
	else 				  cout << "NO\n";
	cout << "Epsilon value:                    " << _objective_epsilon << endl;
  	cout << "Limit to the number of solutions: " << _max_number_of_solutions << endl;
  	cout << "Number of solutions found so far: " << _number_of_solutions << endl;
}//SimpleSolutionManager




