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

const std::vector<std::string>
TokenCon::get_expr_elements_array () {
  
  vector<string> elements;
  for ( auto elem : _exprs ) {
    
    // Make sure to avoid useless '[' or ']'
    char * pch;
    char c_str[ elem.size() + 1 ];
    string tkn_str;
    strcpy(c_str, elem.c_str());
    pch = strtok ( c_str, " ,");
    while ( pch != NULL ) {
      tkn_str = pch;
      
      if ( (tkn_str.at( 0 ) == '[') &&
           (tkn_str.at( tkn_str.length() - 1 ) != ']')) {
        tkn_str = tkn_str. substr ( 1, tkn_str.length() );
      }
      
      if ( (tkn_str.size() >= 3) &&
           (tkn_str.at( 0 ) != '[') &&
           (tkn_str.at( tkn_str.length() - 1 ) == ']') &&
           ((tkn_str.at( tkn_str.length() - 2 ) == ']') ||
            (tkn_str.at( 0 ) >= '1' && tkn_str.at( 0 ) <= '9') )
          ) {
        tkn_str = tkn_str. substr ( 0, tkn_str.length() - 1 );
      }
      
      elements.push_back( tkn_str );
      pch = strtok (NULL, " ,");
    }
  }
  return elements;
}//get_expr_elements_array

const std::vector<std::string>
TokenCon::get_expr_var_elements_array () {
  vector<string> elements;
  for ( auto elem : _exprs ) {
    
    // Make sure to avoid useless '[' or ']'
    char * pch;
    char c_str[ elem.size() + 1 ];
    string tkn_str;
    strcpy(c_str, elem.c_str());
    pch = strtok ( c_str, " ,");
    while ( pch != NULL ) {
      tkn_str = pch;
      
      if ( (tkn_str.at ( 0 ) == '[') &&
           (tkn_str.at ( tkn_str.length() - 1 ) != ']')) {
        tkn_str = tkn_str. substr ( 1, tkn_str.length() );
      }
      
      if ( !((tkn_str.at ( 0 ) >= 'A' && tkn_str.at ( 0 ) <= 'Z') ||
             (tkn_str.at ( 0 ) >= 'a' && tkn_str.at ( 0 ) <= 'z'))  ) {
        pch = strtok ( NULL, " ," );
        continue;
      }
      
      if ( (tkn_str.size() >= 3) &&
           (tkn_str.at ( 0 ) != '[') &&
           (tkn_str.at ( tkn_str.length() - 1 ) == ']') &&
           ((tkn_str.at ( tkn_str.length() - 2 ) == ']') ||
           (tkn_str.at ( 0 ) >= '1' &&
            tkn_str.at ( 0 ) <= '9') )
          ) {
        tkn_str = tkn_str. substr ( 0, tkn_str.length() - 1 );
      }
      
      elements.push_back ( tkn_str );
      pch = strtok ( NULL, " ," );
    }
  }
  return elements;
}//get_expr_var_elements_array

const std::vector<std::string>
TokenCon::get_expr_not_var_elements_array () {
  vector<string> all_elements = get_expr_elements_array ();
  vector<string> all_variable = get_expr_var_elements_array ();
  vector<string> all_non_vars;
  // "Subtract" the second from the first
  for ( auto x: all_elements ) {
    auto iter = std::find( all_variable.begin (), all_variable.end (), x );
    if ( iter == all_variable.end () ) {
      all_non_vars.push_back( *iter );
    }
  }
  return all_non_vars;
}//get_expr_not_var_elements_array

void
TokenCon::print () const {
  cout << "Constraint_" << get_id() << ":\n";
  cout << "Type: " << _con_id << endl;
  cout << "Defined on: ";
  for ( auto x: _exprs ) cout << x << " ";
  cout << "\n";
}//print

