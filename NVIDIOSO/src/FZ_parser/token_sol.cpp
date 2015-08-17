//
//  token_sol.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "token_sol.h"

using namespace std;

TokenSol::TokenSol () :
Token              ( TokenType::FD_SOLVE ),
_var_goal {},
_solve_goal {},
_search_choice {},
_label_choice {},
_variable_choice {},
_assignment_choice {},
_strategy_choice {} {
  _dbg = "TokenSol - ";
}//TokenSol


bool 
TokenSol::set_token ( std::string& token_str )
{
	std::size_t found;
  	found = token_str.find( "satisfy" );
  	if ( found != std::string::npos ) 
  	{
  		set_solve_goal ( "satisfy" );
  	}
  
  	found = token_str.find ( "minimize" );
  	if ( found != std::string::npos ) 
  	{
  		set_solve_goal ( "minimize" );	
    	string str_aux = token_str.substr ( found + 8 );
    	string var_to_minimize = "";
    	for ( int i = 0; i < str_aux.size(); i++ ) 
    	{
    		char x = str_aux[i];
      		if ( x == ' ' ) continue;
      		var_to_minimize += x;
    	}
    	
    	// Sanity check 
    	assert ( var_to_minimize != "" );
    	set_var_goal ( var_to_minimize );
  	}
  
  	found = token_str.find ( "maximize" );
  	if ( found != std::string::npos ) 
  	{	
  		set_solve_goal ( "maximize" );
 	   	string str_aux = token_str.substr ( found + 8 );
    	string var_to_maximize = "";
    	for ( int i = 0; i < str_aux.size(); i++ ) 
    	{
    		char x = str_aux[i];
      		if ( x == ' ' ) continue;
      		var_to_maximize += x;
    	}
    	
    	// Sanity check 
    	assert ( var_to_maximize != "" );
    	set_var_goal ( var_to_maximize );
  	}
  	
  	// Check annotations
  	found = token_str.find( "::" );
  	if ( found != std::string::npos ) 
  	{
  		if ( !set_solve_params ( token_str ) )
  		{
  			LogMsg.error ( _dbg + "Parse error in solve statement: " + 
  			 			   token_str.substr ( found + 2 ), __FILE__, __LINE__);
      		return false;
  		}
    }
	return true;
}//set_token

bool
TokenSol::set_solve_params( string& annotation ) 
{
	vector < string > search_annotation = 
	{
		"int_search",
		"bool_search",
		"set_search"
	};
	
	vector < string > var_choice_annotation =
	{
		"input_order",
		"first_fail",
		"anti_first_fail",
		"smallest",
		"largest",
		"occurence",
		"most_constrained",
		"max_regret"
	};
	
	vector < string > assignment_annotation =
	{
		"indomain_min",
		"indomain_max",
		"indomain_middle",
		"indomain_median",
		"indomain",
		"indomain_random",
		"indomain_split",
		"indomain_reverse_split",
		"indomain_interval"
	};
	
	vector < string > strategy_annotation =
	{
		"complete",
	};
	  
	// search_annotation
	for ( auto& x : search_annotation )
	{
		std::size_t found = annotation.find ( x );
		if ( found != std::string::npos )
		{
			_search_choice = x;
			break;
		}
	}
	
	// Sanity check
	if ( _search_choice == "" )
	{
		return false;	
	}
	
	// var_choice_annotation
	for ( auto& x : var_choice_annotation )
	{
		std::size_t found = annotation.find ( x );
		if ( found != std::string::npos )
		{
			_variable_choice = x;
			break;
		}
	}
	
	// Sanity check
	if ( _variable_choice == "" )
	{
		return false;	
	}
	
	// assignment_annotation
	for ( auto& x : assignment_annotation )
	{
		std::size_t found = annotation.find ( x );
		if ( found != std::string::npos )
		{
			_assignment_choice = x;
			break;
		}
	}
	
	// Sanity check
	if ( _assignment_choice == "" )
	{
		return false;	
	}
	
	// strategy_annotation
	for ( auto& x : strategy_annotation )
	{
		std::size_t found = annotation.find ( x );
		if ( found != std::string::npos )
		{
			_strategy_choice = x;
			break;
		}
	}
	
	// Sanity check
	if ( _strategy_choice == "" )
	{
		return false;	
	}
	
	// Variables to label
	std::size_t brk_found = annotation.find ( "[" );
	std::size_t par_found = annotation.find ( "(" );
	std::size_t com_found = annotation.find ( "," );
	
	// Sanity check 
	assert ( par_found != std::string::npos );
	assert ( com_found != std::string::npos );
	 
	if ( brk_found != std::string::npos )
	{// Array of variables to label
		std::size_t brk_close = annotation.find_last_of ( "]" );
		std::string str_vars = annotation.substr ( par_found + 2, brk_close - par_found - 2 );
		
		// Tokenize vars which are comma separated
		std::istringstream ss ( str_vars );
		std::string tok_var;
		while ( std::getline ( ss, tok_var, ',' ) )
		{
			_var_to_label.push_back ( tok_var );
		}
	}
	else
	{// One single variable (it can be an array of vars)
		std::string str_vars = annotation.substr ( par_found + 1, com_found - par_found - 1 );
		_label_choice = str_vars;
	}
	
	return true;
}//set_solve_goal

void
TokenSol::set_var_goal ( string var_goal ) 
{
	_var_goal = var_goal;
}//set_var_goal

void
TokenSol::set_solve_goal ( string solve ) 
{
	_solve_goal = solve;
}//set_solve_goal

void
TokenSol::set_search_choice ( string search_choice ) 
{
	_search_choice = search_choice;
}//set_search_choice

void
TokenSol::set_label_choice ( string label_choice ) 
{
	_label_choice = label_choice;
}//set_search_choice

void
TokenSol::set_variable_choice ( string var_choice ) 
{
	_variable_choice = var_choice;
}//set_search_choice

void
TokenSol::set_assignment_choice ( string assignment_choice ) 
{
	_assignment_choice = assignment_choice;
}//set_assignment_choice

void
TokenSol::set_strategy_choice ( string strategy_choice ) 
{
	_strategy_choice = strategy_choice;
}//set_strategy_choice

void
TokenSol::set_var_to_label ( string var_to_label ) 
{
  _var_to_label.push_back( var_to_label );
}//set_var_to_label

string
TokenSol::get_var_goal () const {
  return _var_goal;
}//get_solve_goal

string
TokenSol::get_solve_goal () const {
  return _solve_goal;
}//get_solve_goal

string
TokenSol::get_search_choice () const {
  return _search_choice;
}//get_search_choice

std::string
TokenSol::get_label_choice () const {
  return _label_choice;
}//get_label_choice

string
TokenSol::get_variable_choice () const {
  return _variable_choice;
}//get_variable_choice

string
TokenSol::get_assignment_choice () const {
  return _assignment_choice;
}//get_assignment_choice

string
TokenSol::get_strategy_choice () const {
  return _strategy_choice;
}//get_strategy_choice

int
TokenSol::num_var_to_label () const {
  return (int) _var_to_label.size();
}//get_num_var_to_label

vector< std::string >
TokenSol::get_var_to_label () const {
  return _var_to_label;
}//get_var_to_label

string
TokenSol::get_var_to_label ( int idx ) const {
  if ( (idx < 0) ||(idx >= _var_to_label.size()) ) 
  {
	return "";
  }
  return _var_to_label [ idx ];
}//get_var_to_label

void
TokenSol::print () const 
{
	cout << "Solve:\n";
  	cout << "Goal - " << _solve_goal << " ";
  	if ( _var_goal.compare ( "" ) ) 
  	{
    	cout << " on " << _var_goal;
  	}
  	cout << "\n";
  	if ( _search_choice.compare ( "" ) ) 
  	{
    	cout << "Search - " << _search_choice << "\n";
  	}
  	if ( _label_choice.compare ( "" ) ) 
  	{
    	cout << "Label Var - " << _label_choice << "\n";
  	}
  	if ( num_var_to_label () ) 
  	{
    	cout << "On:\n";
    	
    	//for ( auto x : _var_to_label ) {
    	for ( int i = 0; i < _var_to_label.size(); i++ )
    	{
        	auto x = _var_to_label[i];
        	cout << x << " ";
    	}
    	cout << "\n";
  	}
  	if ( _variable_choice.compare ( "" ) ) 
  	{
    	cout << "Variable choice - " << _variable_choice << "\n";
  	}
  	if ( _assignment_choice.compare ( "" ) ) {
    	cout << "Assignment choice - " << _assignment_choice << "\n";
  	}
  	if ( _strategy_choice.compare ( "" ) ) 
  	{
    	cout << "Strategy choice - " << _strategy_choice << "\n";
  	}
}//print



