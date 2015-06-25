//
//  id_generator.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "id_generator.h"

using namespace std;

// Init static variable
IdGenerator* IdGenerator::_id_gen_instance = nullptr;

//Global instance
IdGenerator* glb_id_gen = IdGenerator::get_instance();

IdGenerator::IdGenerator () {
  _str_prefix = "_";
  // Reset counters
  reset_int_id ();
  reset_str_id ();
}//IdGenerator

string
IdGenerator::n_to_str ( int val ) {
  ostringstream convert;
  convert << val;
  return convert.str();
}//n_to_str

void
IdGenerator::reset_int_id () {
  _int_id = 0;
}//reset_int_id

void
IdGenerator::reset_str_id () {
  _str_id = 0;
}//reset_str_id

void
IdGenerator::set_base_offset ( int base ) {
  if ( !_int_id ) {
    _int_id = base;
  }
}//set_base_offset

void
IdGenerator::set_base_prefix ( std::string prefix ) {
  if ( _str_prefix.compare ( "_" ) == 0 ) {
    _str_prefix = prefix;
  }
}//set_base_prefix

int
IdGenerator::get_int_id () {
  return _int_id++;
}//get_int_id

std::string
IdGenerator::get_str_id () {
  return ( _str_prefix + n_to_str( _str_id++ ) );
}//get_str_id

int
IdGenerator::new_int_id () {
  return _int_id++;
}//get_int_id

std::string
IdGenerator::new_str_id () {
  ostringstream convert;
  convert << _str_id++;
  return ( _str_prefix + n_to_str( _str_id++ ) );
}//get_str_id

int
IdGenerator::curr_int_id () {
  return _int_id;
}//curr_int_id

std::string
IdGenerator::curr_str_id () {
  ostringstream convert;
  convert << _str_id;
  return ( _str_prefix + n_to_str( _str_id ) );
}//curr_str_id

void
IdGenerator::print_int_id () {
  cout << "IdGenerator - Current int Id:\t" <<
  _int_id << "\n";
}//print_int_id

void
IdGenerator::print_str_id () {
  cout << "IdGenerator - Current str Id:\t" <<
  _str_prefix + n_to_str( _str_id ) << "\n";
}//print_str_id



