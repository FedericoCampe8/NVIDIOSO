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

bool 
TokenCon::set_token ( std::string& token_str ) 
{
	// Read constraint identifier
  	size_t ptr_idx, ptr_aux;
  	ptr_idx = token_str.find_first_of( "(" );
  	ptr_aux = token_str.find_first_of( ")" );
  	if ( (ptr_idx == std::string::npos) ||
       	 (ptr_aux == std::string::npos) ||
       	 (ptr_aux < ptr_idx) ) 
    {
    	LogMsg.error( _dbg + "Constraint not valid" + token_str,
         	          __FILE__, __LINE__ );
    	return false;
  	}
  
  	set_con_id( token_str.substr( 0, ptr_idx ) );

  	// Get the expressions that identify the constraint
  	token_str = token_str.substr ( ptr_idx + 1, ptr_aux - ptr_idx - 1 );
  
  	int brk_counter = 0;
  	string expression = "";
  	for ( int i = 0; i < token_str.size(); i++ ) 
  	{
  		char x = token_str[i];	
    	if ( x == '[' ) brk_counter++;
    	if ( x == ']' ) 
    	{
      		expression += x;
      		if ( brk_counter ) 
      		{
        		brk_counter--;
        		if ( brk_counter == 0 ) 
        		{
        			add_expr ( expression );
		          	expression.assign("");
        		}
      		}
    	}//']'
    	else if ( x == ',' ) 
    	{
      		if ( brk_counter > 0 ) 
      		{
        		expression += x;
      		}
      		else if ( expression.length() ) 
      		{
      			add_expr ( expression );
	        	expression.assign("");
      		}
    	}
    	else if ( x == ' ' ) 
    	{
      		if ( brk_counter > 0 ) {
        		expression += x;
      		}
      		else 
      		{
        		expression.assign("");
      		}
    	}
    	else 
    	{
      	expression += x;
    	}
  	}//x
  	if ( expression.length() ) 
  	{
  		add_expr ( expression );
  	}
  	return true;
}//set_token

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
  //for ( auto elem : _exprs ) {
  for ( int ii = 0; ii < _exprs.size(); ii++ )
  {
      auto elem = _exprs[ii];
      
    // Make sure to avoid useless '[' or ']'
    char * pch;
    char c_str[ elem.size() + 1 ];
    string tkn_str;
    strcpy(c_str, elem.c_str());
    pch = strtok ( c_str, " ,");
    
    while ( pch != NULL ) {
      tkn_str = pch;
      
      if ( tkn_str.at( 0 ) == '[' /*&&
           (tkn_str.at( tkn_str.length() - 1 ) != ']')*/) {
        tkn_str = tkn_str. substr ( 1, tkn_str.length() );
      }
      
      if ( (tkn_str.size() >= 3) &&
           (tkn_str.at( 0 ) != '[') &&
           (tkn_str.at( tkn_str.length() - 1 )  == ']') &&
           ((tkn_str.at( tkn_str.length() - 2 ) == ']') ||
            ((tkn_str.at( 0 ) >= '1' && tkn_str.at( 0 ) <= '9') ||
             (tkn_str.at( 0 ) == '+' || tkn_str.at( 0 ) == '-') ))
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
  //for ( auto elem : _exprs ) {
  for ( int ii = 0; ii < _exprs.size(); ii++ )
  {
      auto elem = _exprs[ii];
    // Make sure to avoid useless '[' or ']'
    char * pch;
    char c_str[ elem.size() + 1 ];
    string tkn_str;
    strcpy(c_str, elem.c_str());
    pch = strtok ( c_str, " ,");
    while ( pch != NULL ) {
      tkn_str = pch;
      
      if ( tkn_str.at ( 0 ) == '[' /* &&
           (tkn_str.at ( tkn_str.length() - 1 ) != ']')*/) {
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
  //for ( auto x: all_elements ) {
  for ( int i = 0; i < all_elements.size(); i++  )
  {
      auto x = all_elements[i];
      auto iter = std::find( all_variable.begin (), all_variable.end (), x );
      if ( iter == all_variable.end () )
          all_non_vars.push_back( x );
  }

  return all_non_vars;
}//get_expr_not_var_elements_array

void
TokenCon::print () const {
  cout << "Constraint_" << get_id() << ":\n";
  cout << "Type: " << _con_id << endl;
  cout << "Defined on: ";
  //for ( auto x: _exprs ) cout << x << " ";
  for ( int i = 0; i < _exprs.size(); i++ )
      cout << _exprs[i] << " ";
  cout << "\n";
}//print

