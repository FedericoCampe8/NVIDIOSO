//
//  token_con.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token_con.h"

using namespace std;

std::vector<std::string> TokenCon::global_constraint_keyword = 
{
	"abs_value",
	"all_differ_from_at_least_k_pos",
	"all_differ_from_at_most_k_pos",
	"all_differ_from_exactly_k_pos",
	"all_equal",
	"all_equal_peak",
	"all_equal_peak_max",
	"all_equal_valley",
	"all_equal_valley_min",
	"all_incomparable",
	"all_min_dist",
	"alldifferent",
	"alldifferent_between_sets",
	"alldifferent_consecutive_values",
	"alldifferent_cst",
	"alldifferent_except_0",
	"alldifferent_interval",
	"alldifferent_modulo",
	"alldifferent_on_intersection",
	"alldifferent_partition",
	"alldifferent_same_value",
	"allperm",
	"among",
	"among_diff_0",
	"among_interval",
	"among_low_up",
	"among_modulo",
	"among_seq",
	"among_var",
	"and",
	"arith",
	"arith_or",
	"arith_sliding",
	"assign_and_counts",
	"assign_and_nvalues",
	"atleast",
	"atleast_nvalue",
	"atleast_nvector",
	"atmost",
	"atmost_nvalue",
	"atmost_nvector",
	"balance",
	"balance_cycle",
	"balance_interval",
	"balance_modulo",
	"balance_partition",
	"balance_path",
	"balance_tree",
	"between_min_max",
	"big_peak",
	"big_valley",
	"bin_packing",
	"bin_packing_capa",
	"binary_tree",
	"bipartite",
	"calendar",
	"cardinality_atleast",
	"cardinality_atmost",
	"cardinality_atmost_partition",
	"change",
	"change_continuity",
	"change_pair",
	"change_partition",
	"change_vectors",
	"circuit",
	"circuit_cluster",
	"circular_change",
	"clause_and",
	"clause_or",
	"clique", 
	
	/* 5.71 in http://sofdem.github.io/gccat/gccat/Catleast.html */
	/* core global constraints here */
	
	"cumulative",
	"cycle",
	"diffn",
	"disjunctive",
	"element",
	"global_cardinality",
	"global_cardinality_with_costs",
	"minimum_weight_alldifferent",
	"nvalue",
	"sort"
};

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
  
  	std::string constraint_name = token_str.substr( 0, ptr_idx );
  	set_con_id ( constraint_name );
	
	if ( global_constraint ( constraint_name ) )
	{
		set_type ( TokenType::FD_GLB_CONSTRAINT );
	}
	
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

bool 
TokenCon::global_constraint ( std::string constraint_name )
{
	auto it = std::find ( 
	TokenCon::global_constraint_keyword.begin(), 
	TokenCon::global_constraint_keyword.end(),
	constraint_name );
	if ( it != TokenCon::global_constraint_keyword.end () ) 
	{
		return true;
	}
	
	return false;
}//global_constraint

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

