//
//  cp_variable.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "variable.h"

using namespace std;

Variable::Variable () :
_str_id     ( "" ),
_var_type   ( VariableType::OTHER ),
_domain_ptr ( nullptr ) {
  _id = glb_id_gen->get_int_id();
}//Variable

Variable::Variable ( int id ) :
_id         ( id ),
_str_id     ( "" ),
_var_type   ( VariableType::OTHER ),
_domain_ptr ( nullptr ){
}//Variable

Variable::~Variable () {
}//~Variable

int
Variable::get_id () const {
  return _id;
}//get_id

void
Variable::set_str_id ( string str_id ) {
  if ( _str_id.compare( "" ) == 0 ) {
    _str_id = str_id;
  }
}//set_str_id

string
Variable::get_str_id () const {
  return _str_id;
}//get_str_id

void
Variable::set_type ( VariableType v_type ) {
  if ( _var_type == VariableType::OTHER ) {
    _var_type = v_type;
  }
}//set_type

VariableType
Variable::get_type () const {
  return _var_type;
}//get_type

void
Variable::set_domain ( DomainType  dt ) {
  _domain_ptr->set_type ( dt );
}//set_domain

void
Variable::print () const  {
  cout << "Variable_"   << _id     << "\n";
  cout << "Variable id: \"" << _str_id << "\"\n";
  cout << "Type:\t";
  switch ( _var_type ) {
    case VariableType::FD_VARIABLE:
      cout << "FD_Var\n";
      break;
    case VariableType::SUP_VARIABLE:
      cout << "SUP_Var\n";
      break;
    case VariableType::OBJ_VARIABLE:
      cout << "OBJ_Var\n";
      break;
    default:
      cout << "Not Defined\n";
  }
  
}//print



