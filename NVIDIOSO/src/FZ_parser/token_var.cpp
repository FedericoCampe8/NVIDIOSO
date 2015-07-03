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
    Token         ( TokenType::FD_VARIABLE ),
    _var_id       ( "" ),
    _support_var  ( false ),
    _lw_bound     ( 0  ),
    _up_bound     ( -1 ),
    _var_dom_type ( VarDomainType::OTHER ) {
    _dbg = "TokenVar - ";
}//TokenVar

TokenVar::~TokenVar () {
}//~TokenVar

void
TokenVar::set_subset_domain ( std::string token_str )
{
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
TokenVar::get_range ( std::string token_str ) const {
    size_t ptr, ptr_aux;
    pair<int, int> range;
    range.first  =  0;
    range.second = -1;
  
    if ( token_str.find( ".." ) != std::string::npos ) {
        ptr = token_str.find_first_of( ".", 0 );
    
        // Clear string
        ptr_aux = ptr;
        while ( (token_str[ ptr_aux ] != ' ') && (ptr_aux != 0) ) ptr_aux--;
        if ( ptr_aux != 0 ) ptr_aux++;
    
        range.first = atoi( token_str.substr( ptr_aux, ptr ).c_str() );
        ptr = token_str.find_first_of( ":", ptr+2 );

        //Check ' ' before ':'
        token_str = token_str.substr( token_str.find_first_of( ".", 0 ) + 2, ptr );
        ptr = token_str.find_first_of( " ", 0 );
        if ( ptr == std::string::npos )
        {
            range.second = atoi ( token_str.c_str() );
        }
        else
        {
            range.second = atoi ( token_str.substr( 0, ptr ).c_str() );
        }
    }
    else
    {
        LogMsg.error( _dbg + "Domain range not valid", __FILE__, __LINE__ );
    }
    return range;
}//get_range

vector<int>
TokenVar::get_subset ( std::string token_str ) const {
  
  char * pch;
  char * c_str = new char[ token_str.length() + 1 ];
  vector <int> domain_values;
  
  // Consistency check
  size_t find = token_str.find ( "{" );
  if ( find == string::npos ) { return domain_values; }
  
  token_str = token_str.substr ( find );
  strncpy ( c_str, token_str.c_str(), token_str.length() );
  
  pch = strtok ( c_str, " {,:" );
  while ( pch != NULL ) {
    int pos = 0;
    bool find_close_brk = false;
    for ( ; pch[ pos ] != '\0'; pos++ ) {
      if ( pch[ pos ] == '}' ) {
        pch[ pos ] = '\0';
        find_close_brk = true;
        break;
      }
    }
    domain_values.push_back( atoi( pch ) );
    if ( find_close_brk ) break;
    
    pch = strtok ( NULL, " {,:" );
  }
  delete [] c_str;
  
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
TokenVar::set_objective_var () {
  _objective_var = true;
  set_support_var ();
  set_int_domain  ();
}//set_objective_var

bool
TokenVar::is_objective_var () const {
  return _objective_var;
}//is_objective_var

void
TokenVar::set_support_var () {
  _support_var = true;
}//set_support_var

bool
TokenVar::is_support_var () const {
  return _support_var;
}//is_support_var

void
TokenVar::set_var_dom_type ( VarDomainType vdt ) {
  if ( _var_dom_type == VarDomainType::OTHER ) {
    _var_dom_type = vdt;
  }
}//set_var_dom_type

VarDomainType
TokenVar::get_var_dom_type () const {
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
TokenVar::set_range_domain ( int lw_b, int up_b ) {
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
TokenVar::get_lw_bound_domain () const {
  return _lw_bound;
}//get_lw_bound_domain

int
TokenVar::get_up_bound_domain () const {
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
TokenVar::set_subset_domain ( const vector < vector <  int > >& elems ) {
    // The following is not recognized by gcc < 47
    //for ( auto x : elems ) set_subset_domain ( x );
    for ( int i = 0; i < elems.size(); i++ )
        set_subset_domain ( elems[i] );
}//set_subset_domain

vector < vector< int> >
TokenVar::get_subset_domain () {
  return _subset_domain;
}//get_set_domain

void
TokenVar::set_subset_domain ( const std::pair <int, int>& range ) {
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
