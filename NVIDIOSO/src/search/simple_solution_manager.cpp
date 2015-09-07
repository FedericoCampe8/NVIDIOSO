//
//  simple_solution_manager.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_solution_manager.h"

using namespace std;

SimpleSolutionManager::SimpleSolutionManager () :
_find_all_solutions      ( false ),
_max_number_of_solutions ( 1 ) ,
_number_of_solutions     ( 0 ) {
}//SimpleSolutionManager

SimpleSolutionManager::SimpleSolutionManager ( std::vector < Variable* >& vars ) :
SimpleSolutionManager() {
  if ( vars.size() )
    for ( auto var : vars )
      _variables[ var->get_id() ] = var;
}//SimpleSolutionManager

SimpleSolutionManager::~SimpleSolutionManager () {
  _variables.clear();
  _solution_strings.clear();
}//~SimpleSolutionManager

void
SimpleSolutionManager::set_variables ( std::vector < Variable* >& vars ) {
  if ( !vars.size() ) return;
  
  for ( auto var : vars )
    _variables[ var->get_id() ] = var;
}//set_variables

void
SimpleSolutionManager::print_solution () {
  if ( !_solution_strings.size ()) return;
  
  cout << get_solution() << endl;
}//print_solution

size_t
SimpleSolutionManager::number_of_solutions () {
  return _number_of_solutions;
}//number_of_solutions
 
std::string
SimpleSolutionManager::get_solution () const {
  if ( !_solution_strings.size () ) return "";
  
  return _solution_strings[ _solution_strings.size () - 1  ];
}//get_solution

std::string
SimpleSolutionManager::get_solution ( size_t sol_idx ) const {
  if ( sol_idx < 1 || sol_idx > _number_of_solutions ) {
    return "";
  }
  return  _solution_strings[ sol_idx - 1 ];
}//get_solution

std::vector< std::string >
SimpleSolutionManager::get_all_solutions () const {
  return _solution_strings;
}//get_all_solutions

void
SimpleSolutionManager::set_solution_limit ( int n_sol ) {
  if ( n_sol < 0 ) 
  { 
	_find_all_solutions = true;
    return;
  }
  _find_all_solutions      = false;
  _max_number_of_solutions = n_sol;
}//set_solution_limit

bool
SimpleSolutionManager::notify () 
{
	if ( !_variables.size() ) return true;
  
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
  
  	if ( _find_all_solutions ) return false;
  	if ( _number_of_solutions >= _max_number_of_solutions ) return true;
  
  	return false;
}//notify

void
SimpleSolutionManager::print_variables () {
  for ( auto var : _variables ) var.second->print_domain ();
}//print_variables

void
SimpleSolutionManager::print () const {
  cout << "Simple Solution Manager:\n";
  cout << "Limit to the number of solutions: " << _max_number_of_solutions << endl;
  cout << "Number of solutions found so far: " << _number_of_solutions << endl;
}//SimpleSolutionManager




