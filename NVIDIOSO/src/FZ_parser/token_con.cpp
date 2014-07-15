//
//  token_con.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token_con.h"

using namespace std;

TokenCon::TokenCon () :
Token         ( TokenType::FD_CONSTRAINT ),
_con_id       ( "" ) {
  _dbg = "TokenCon - ";
}//TokenVar

void
TokenCon::set_con_id ( string con_id ) {
  if ( !_con_id.compare( "" ) ) {
    _con_id = con_id;
  }
}//set_con_id

string
TokenCon::get_con_id () const {
  return _con_id;
}//get_con_id

void 
TokenCon::add_expr ( string expr ) {
  _exprs.push_back( expr );
}//add_expr

int
TokenCon::get_num_expr () const {
  return (int) _exprs.size();
}//get_num_expr

std::string
TokenCon::get_expr ( int idx ) const {
  if ( (idx < 0) || (idx >= _exprs.size()) ) return "";
  return _exprs[ idx ];
}//get_expr

const std::vector<std::string>
TokenCon::get_expr_array () {
  return _exprs;
}//get_expr_array

void
TokenCon::print () const {
  cout << "Constraint_" << get_id() << ":\n";
  cout << "Type: " << _con_id << endl;
  cout << "Defined on: ";
  for ( auto x: _exprs ) cout << x << " ";
  cout << "\n";
}//print

