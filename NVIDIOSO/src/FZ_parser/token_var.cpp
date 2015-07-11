//
//  token_var.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 05/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token_var.h"

using namespace std;

TokenVar::TokenVar () :
    Token          ( TokenType::FD_VARIABLE ),
    _var_id        ( "" ),
    _objective_var ( false ),
    _support_var   ( false ),
    _lw_bound      ( 0  ),
    _up_bound      ( -1 ),
    _var_dom_type  ( VarDomainType::OTHER ) {
    _dbg = "TokenVar - ";
}//TokenVar

TokenVar::~TokenVar () {
}//~TokenVar

bool 
TokenVar::set_token ( std::string& token_str )
{
	// Check whether the variable is a support variable
  	if ( token_str.find ( ":: var_is_introduced") != string::npos ) 
  	{
  		set_support_var();
  	}
  	
  	std::string var_str = token_str;
  	std::size_t idx = var_str.find ( "::" ); 
	if ( idx != std::string::npos )
	{
		var_str = var_str.substr ( 0, idx - 1 );
		int idx_last = var_str.size() - 1;
		while ( var_str[ idx_last ] == ' ' ) idx_last--;
		var_str [ idx_last + 1 ] = ';';
	}
	
  	
  	// Set variable type
  	set_type_var ( var_str ); //token_str
  	
  	// Set variable id
  	set_id ( var_str ); //token_str
  	
  	// Set objective var
  	if ( _var_id == "fobj" ) 
  	{
  		set_objective_var ();
  	}
  	
  	return true;
}//set_token

bool
TokenVar::set_type_var ( std::string& type_str )
{
	// var_type
	vector < string > type =
	{
		"bool",
		"float",
		"int",
		"..",	// Range domain type
		"{"		// Set domain type
	};
	
	bool set_type = false;
	for ( auto& x : type )
	{
		std::size_t found = type_str.find ( x );
		if ( found != std::string::npos && x == "bool" )
		{// [1..20] of var bool: q;
			set_boolean_domain ();
			set_type = true;
			break;
		}
		if ( found != std::string::npos && x == "float" )
		{// [1..20] of var float: q;
			set_float_domain ();
			set_type = true;
			break;
		}
		if ( found != std::string::npos && x == "int" )
		{// [1..20] of var int: q; or [1..20] of var set of int
			std::size_t found_set = type_str.find ( "set" );
			if ( found_set != std::string::npos )
			{
				set_subset_domain ( type_str );
			}
			else
			{
				set_int_domain ();
			}
			set_type = true;
			break;
		}
		if ( found != std::string::npos && x == ".." )
		{// [1..20] of var 1..20: q; or [1..20] of set of 1..20: q;
			std::size_t found_set = type_str.find ( "set" );
			if ( found_set != std::string::npos )
			{
				set_subset_domain ( type_str );
			}
			else
			{
				set_range_domain ( type_str );
			}
			set_type = true;
			break;
		}
		if ( found != std::string::npos && x == "{" )
		{// [1..20] of var {1,2,3}: q; or [1..20] of set of {1,2,3}: q;
			set_subset_domain ( type_str );
			set_type = true;
			break;
		}
	}
	
	if ( !set_type ) 
		return false;
	
	return true;
}//set_type_var

void 
TokenVar::set_id ( std::string& id_str ) 
{
	// identifier
	std::size_t idx = id_str.find_last_of ( ":" );
	
	// Sanity check
	assert ( idx != std::string::npos );
	
	std::string id_ann_str = id_str.substr ( idx + 1 );
	
	istringstream iss ( id_ann_str );
	vector<string> id_ann_tokens { istream_iterator<string>{iss},
                      	    	   istream_iterator<string>{} };
                      	    
	// Sanity check
	assert ( id_ann_tokens.size() > 0 );
	
	if ( id_ann_tokens [ 0 ][id_ann_tokens [ 0 ].size() - 1] == ';' )
	{
		id_ann_tokens [ 0 ] = id_ann_tokens [ 0 ].substr ( 0, id_ann_tokens [ 0 ].size() - 1);
	}
	
	set_var_id ( id_ann_tokens [ 0 ] );
}//set_id

void
TokenVar::set_subset_domain ( std::string token_str )
{
	// It can be [1..20] of set of 1..20: q; or [1..20] of var (set of) {1,2,3}: q;
    if ( token_str.find ( "int" ) != std::string::npos )
    {
        set_subset_domain ();
    }
    else if ( token_str.find ( ".." ) != std::string::npos )
    {
        set_subset_domain ( get_range ( token_str ) );
    }
    else if ( token_str.find ( "{" ) != std::string::npos )
    {
        set_subset_domain ( get_subset ( token_str ) );
    }
    else
    {
        LogMsg.error( _dbg + "Parse Error in variable declaration: " + token_str,
                       __FILE__, __LINE__);
        set_subset_domain ();
    }
}//set_subset_domain


pair<int, int>
TokenVar::get_range ( std::string token_str ) const 
{
	pair<int, int> range = { 0, -1 };

	std::size_t found = token_str.find( ".." );
	if ( found == std::string::npos )
	{
		 LogMsg.error( _dbg + "Range not found: " + token_str, __FILE__, __LINE__ );
		 return range;
	}
	
	std::size_t start = found - 1;
	while ( start > 0 && (token_str [ start ] >= '0' && token_str [ start ] <= '9' ) )
	{
		start--;
	}
	// Start is either 0 or a non valid char
	if ( !(token_str [ start ] >= '0' && token_str [ start ] <= '9') )
	{
		start++;
	}
	
	std::size_t end = found + 2;
	while ( end < token_str.size() -1 && (token_str [ end ] >= '0' && token_str [ end ] <= '9' ) )
	{
		end++;
	}
	// End is either the last digit or a non valid digit
	if ( !(token_str [ end ] >= '0' && token_str [ end ] <= '9') )
	{
		end--;
	}
	
	token_str.assign ( token_str.begin() + start,  token_str.begin() + end + 1 );
	
	bool lower_set = false;
	int lower_bound = 0, upper_bound = 0;
	for ( auto& x : token_str )
	{
		if ( x != '.' && !lower_set )
		{
			lower_bound = lower_bound * 10 + static_cast<int> ( x - '0' );
		}
		else if ( x == '.' )
		{
			lower_set = true;
		}
		else
		{
			upper_bound = upper_bound * 10 + static_cast<int> ( x - '0' );
		}
	}
	
	range.first  = lower_bound;
	range.second = upper_bound;
    return range;
}//get_range

vector<int>
TokenVar::get_subset ( std::string token_str ) const 
{
	vector <int> domain_values;
	std::size_t start = token_str.find_first_of ( "{" );
	std::size_t end   = token_str.find_last_of  ( "}" );
  	
  	//Sanity check
  	if ( start == std::string::npos || end == std::string::npos )
  	{
  		LogMsg << _dbg << "Subset for variable not valid: " << token_str << std::endl;
  		return domain_values;
  	}
  	
  	token_str.assign ( token_str.begin() + start, token_str.begin() + end + 1 );
  	std::istringstream ss ( token_str );
  	std::string token_val;

	while ( std::getline (ss, token_val, ',') ) 
	{
		int num;
		istringstream ( token_val ) >> num;
		domain_values.push_back ( num );
	}
  	
  	return domain_values;
}//get_subset

void
TokenVar::set_var_id ( string id ) {
  if ( !_var_id.compare ( "" ) ) {
    _var_id = id;
  }
}//set_var_id

string
TokenVar::get_var_id () const {
  return _var_id;
}//get_var_id

void
TokenVar::set_objective_var () 
{
  _objective_var = true;
  set_support_var ();
  set_int_domain  ();
}//set_objective_var

bool
TokenVar::is_objective_var () const 
{
  return _objective_var;
}//is_objective_var

void
TokenVar::set_support_var () 
{
  _support_var = true;
}//set_support_var

bool
TokenVar::is_support_var () const 
{
  return _support_var;
}//is_support_var

void
TokenVar::set_var_dom_type ( VarDomainType vdt ) {
  if ( _var_dom_type == VarDomainType::OTHER ) 
  {
    _var_dom_type = vdt;
  }
}//set_var_dom_type

VarDomainType
TokenVar::get_var_dom_type () const 
{
  return _var_dom_type;
}//get_var_dom_type

void
TokenVar::set_boolean_domain () {
  _var_dom_type = VarDomainType::BOOLEAN;
}//set_boolean_domain

void
TokenVar::set_float_domain () {
  _var_dom_type = VarDomainType::FLOAT;
}//set_float_domain

void
TokenVar::set_int_domain () {
  _var_dom_type = VarDomainType::INTEGER;
}//set_int_domain

void
TokenVar::set_range_domain ( std::string str ) {
  pair<int, int> range = get_range( str );
  set_range_domain( range.first, range.second );
}//set_range_domain

void
TokenVar::set_range_domain ( int lw_b, int up_b ) 
{
  // Check consistency of bounds
  bool valid = true;
  if ( up_b < lw_b ) {
    LogMsg.error( _dbg + "Domain bounds not valid", __FILE__, __LINE__ );
    valid = false;
  }
  _lw_bound = lw_b;
  if ( !valid ) _up_bound = lw_b;
  else          _up_bound = up_b;

  // Set domain type
  _var_dom_type = VarDomainType::RANGE;
}//set_range_domain

int
TokenVar::get_lw_bound_domain () const 
{
  return _lw_bound;
}//get_lw_bound_domain

int
TokenVar::get_up_bound_domain () const 
{
  return _up_bound;
}//get_up_bound_domain

void
TokenVar::set_subset_domain () {
  _var_dom_type = VarDomainType::SET_INT;
}//set_subset_domain

void
TokenVar::set_subset_domain ( const vector <int>& dom_vec ) {
  _subset_domain.push_back ( dom_vec );
  
  // Set domain type
  _var_dom_type = VarDomainType::SET;
}//set_set_domain

void
TokenVar::set_subset_domain ( const vector < vector < int > >& elems ) {
    // The following is not recognized by gcc < 47
    //for ( auto x : elems ) set_subset_domain ( x );
    for ( int i = 0; i < elems.size(); i++ )
        set_subset_domain ( elems[i] );
}//set_subset_domain

vector < vector< int> >
TokenVar::get_subset_domain () 
{
  return _subset_domain;
}//get_set_domain

void
TokenVar::set_subset_domain ( const std::pair <int, int>& range ) 
{
  set_range_domain ( range.first, range.second );
  
  // Set domain type
  _var_dom_type = VarDomainType::SET_RANGE;
}//set_subset_domain

void
TokenVar::print () const {
  cout << "FD_Var_" << get_id() << ":\n";
  cout << "ID " << _var_id << endl;
  cout << "Support variable: ";
  if ( _support_var ) cout << "T\n";
  else                cout << "F\n";
  cout << "DOMAIN: ";
  switch ( _var_dom_type ) {
    case VarDomainType::BOOLEAN:
      cout << "bool\n";
      break;
    case VarDomainType::FLOAT:
      cout << "float\n";
      break;
    case VarDomainType::INTEGER:
      cout << "int\n";
      break;
    case VarDomainType::RANGE:
      cout << "range " << _lw_bound << ".." << _up_bound << "\n";
      break;
    case VarDomainType::SET:
      cout << "set of { ";
      //for ( auto x: _subset_domain ) {
      for ( int i = 0; i < _subset_domain.size(); i++ )
      {
          auto x = _subset_domain[i];
          cout << "{ ";
          //for ( auto y: x ) cout << y << " ";
          for ( int ii = 0; ii < x.size(); ii++ )
              cout << x[ii] << " ";
          cout << "} ";
      }
      cout << "}\n";
      break;
    case VarDomainType::SET_RANGE:
      cout << "set of ";
      cout << _lw_bound << ".." << _up_bound << "\n";
      //for ( auto x: _subset_domain ) {
      for ( int i = 0; i < _subset_domain.size(); i++ )
      {
          auto x = _subset_domain[i];
          cout << "{ ";
          //for ( auto y: x ) cout << y << " ";
          for ( int ii = 0; ii < x.size(); ii++ )
              cout << x[ii] << " ";
          cout << "} ";
      }
      break;
    case VarDomainType::SET_INT:
      cout << "set of int\n";
      for ( int i = 0; i < _subset_domain.size(); i++ )
      {
          auto x = _subset_domain[i];
          cout << "{ ";
          //for ( auto y: x ) cout << y << " ";
          for ( int ii = 0; ii < x.size(); ii++ )
              cout << x[ii] << " ";
          cout << "} ";
      }
      break;
    default:
    case VarDomainType::OTHER:
      cout << "Other (not specified/recognized)\n";
      break;
  }
  if ( _objective_var ) cout << "Objective Variable\n";
}//print
