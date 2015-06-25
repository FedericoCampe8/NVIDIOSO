//
//  token_arr.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 05/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token_arr.h"

using namespace std;

TokenArr::TokenArr () :
_size           ( -1 ),
_array_lwb      ( 0 ),
_array_upb      ( -1 ),
_lw_var_glb_idx ( 0 ),
_up_var_glb_idx ( 0 ),
_output_arr     ( false ),
_support_array  ( false ),
_support_elements ( "" ) {
  _dbg = "TokenArr - ";
  set_type ( TokenType::FD_VAR_ARRAY );
}//TokenArr

void
TokenArr::set_size_arr ( int size ) {
  if ( _array_lwb > _array_upb ) {
    _size = size;
  }
}//set_size_arr

int
TokenArr::get_size_arr () const {
  return _size;
}//get_size_arr

void
TokenArr::set_array_bounds ( int lwb, int upb ) {
  bool valid = true;
  if ( upb < lwb ) {
    logger->error( _dbg + "Array bounds not valid", __FILE__, __LINE__ );
    valid = false;
  }
  if ( _array_lwb > _array_upb ) {
    // Bounds as read from file
    _array_lwb = lwb;
    if ( !valid ) _array_upb = lwb;
    else          _array_upb = upb;
    // Size
    _size      = upb - lwb + 1;
    // Set lw-up bounds w.r.t. the global ids
    _lw_var_glb_idx = glb_id_gen->curr_int_id ();
    _up_var_glb_idx = _lw_var_glb_idx + _size - 1;
  }
}//set_array_bounds

int
TokenArr::get_lw_bound () const {
  return _array_lwb;
}//get_lw_bound

int
TokenArr::get_up_bound () const {
  return _array_upb;
}//get_up_bound

int
TokenArr::get_lower_var () const {
  return _lw_var_glb_idx;
}//get_lower_var_within

int
TokenArr::get_upper_var () const {
  return _up_var_glb_idx;
}//get_upper_var_within

void
TokenArr::set_output_arr () {
  _output_arr = true;
}//set_output_arr

bool
TokenArr::is_output_arr () const {
  return _output_arr;
}//is_output_arr


bool
TokenArr::is_var_in ( int var_idx ) const {
  return ( (var_idx >= _lw_var_glb_idx) && (var_idx <= _up_var_glb_idx) );
}//is_var_in

bool
TokenArr::is_var_in ( string var_id ) const {
  // Check whether var_id is a valid id for an array
  size_t found     = var_id.find_first_of( "[" );
  size_t found_aux = var_id.find_first_of( "]" );
  if ( (found     == std::string::npos) ||
       (found_aux == std::string::npos) ) return false;
  // Check whether var_id is the id for this array
  string array_id = var_id.substr( 0, found );
  if ( _var_id.compare ( array_id ) ) return false;
  // Check whether the element is within the array
  array_id = var_id.substr( found + 1, found_aux - found - 1 );
  int idx = atoi ( array_id.c_str() );
  return ( (idx >= _array_lwb) && (idx <= _array_upb) );
}//is_var_in

void
TokenArr::set_support_elements ( std::string elem_str ) {
  _support_array    = true;
  _support_elements = elem_str;
}//set_support_elements

string
TokenArr::get_support_elements () const {
  return _support_elements;
}//get_support_elements

void
TokenArr::print () const {
  cout << "FD_Arr_Var_" << get_id() << ":\n";
  cout << "ID " << _var_id;
  cout << " [" <<  _array_lwb << ".." << _array_upb << "]";
  cout << " - Global" <<
  "[" << _lw_var_glb_idx << ".." << _up_var_glb_idx << "]\n";
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
      cout << "range: " << _lw_bound << ".." << _up_bound << "\n";
      break;
    case VarDomainType::SET:
      cout << "set of { ";
      //for ( auto x: _subset_domain ) {
      for ( int i = 0; i < _subset_domain.size(); i++ )
      {
          auto x = _subset_domain[i];
          cout << "{";
          //for ( auto y: x ) cout << y << " ";
          for ( int ii = 0; ii < x.size(); ii++ )
              cout << x[ii] << " ";
          cout << "} ";
      }
      cout << "}\n";
      break;
    default:
    case VarDomainType::OTHER:
      cout << "Other (not specified/recognized)\n";
      break;
  }
  if ( _output_arr )    cout << "Output array\n";
}//print

