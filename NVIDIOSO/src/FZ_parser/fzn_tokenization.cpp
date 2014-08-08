//
//  fzn_tokenization.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 04/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "fzn_tokenization.h"
#include "token_var.h"
#include "token_arr.h"
#include "token_con.h"
#include "token_sol.h"

using namespace std;

FZNTokenization::FZNTokenization () {
  _dbg = "FZNTokenization - ";
  set_delimiter( "\n" );
}//FZNTokenization

FZNTokenization::~FZNTokenization () {
}//~FZNTokenization

TokenPtr
FZNTokenization::get_token () {
  // Init internal state of the tokenizer
  if ( _c_token == nullptr ) {
    _c_token = strtok ( _parsed_line , DELIMITERS.c_str() );
  }
  else { 
    /*
     * The line is not completely parsed:
     * get next token.
     */
    _c_token = strtok ( NULL , DELIMITERS.c_str() );
  }
  // Clear _c_token
  clear_line();
  
  /*
   * If the string is terminated: prepare next one.
   * @note: we do not set _more_elements to false.
   * This means that the client will try again
   * to check whether there are other lines to read
   * from the file.
   */
  if ( _c_token == NULL || skip_line() ) {
    _c_token   = nullptr;
    _need_line = true;
    return nullptr;
  }
  
  TokenPtr current_token = analyze_token ();
  if ( current_token == nullptr ) {
    throw  NvdException ( ( _dbg + "Error in tokenization" ).c_str(),
                          __FILE__, __LINE__ );
  }
  
  return current_token;
}//get_token

TokenPtr
FZNTokenization::analyze_token () {
  
  // Analyze _c_token
  string token_str;
  token_str.assign( _c_token );
  
  // Check if a keyword has been found
  size_t found_arr = token_str.find ( ARR_TOKEN );
  size_t found_con = token_str.find ( CON_TOKEN );
  size_t found_sol = token_str.find ( SOL_TOKEN );
  size_t found_var = token_str.find ( VAR_TOKEN );
  size_t found_lns = token_str.find ( LNS_TOKEN );
  
  if ( found_arr != std::string::npos ) {
    return analyze_token_arr ();
  }
  if ( found_con != std::string::npos ) {
    return analyze_token_con ();
  }
  if ( found_sol != std::string::npos ) {
    return analyze_token_sol ();
  }
  if ( found_var != std::string::npos ) {
    return analyze_token_var ();
  }
  if ( found_lns != std::string::npos ) {
    return analyze_token_lns ();
  }
  
  logger->error( _dbg + "other types not yet implemented: " + token_str,
                 __FILE__, __LINE__ );
  
  return nullptr;
}//analyze_token

TokenPtr
FZNTokenization::analyze_token_arr () {
  
  // Token (pointer) to return
  TokenPtr tkn_ptr = make_shared<TokenArr> ();
  
  // Clear line: avoid useless chars
  for ( int i = 0; i < ARR_TOKEN.length(); i++ ) _c_token++;
  clear_line();
  
  // Use string methods
  string token_str;
  token_str.assign( _c_token );
  
  // Check whether the array is a support variable
  if ( token_str.find( VII_TOKEN, 0 ) != string::npos ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_support_var ();
  }
  
  // Check whether the array is an output array
  if ( token_str.find( OAR_TOKEN, 0 ) != string::npos ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_output_arr ();
  }

  /*
   * Read var_type: differentiate between
   * Parameter type and Variable type.
   */
  size_t ptr, ptr_aux;
  ptr = token_str.find ( "var" );
  if ( ptr != string::npos ) {
    // Variable type
    ptr_aux = ptr + 3;
    ptr     = token_str.find_first_of( ":", 0 );
    ptr_aux = token_str.find_first_not_of ( " ", ptr_aux );
    while ( (ptr > 0) && (token_str.at(ptr - 1 ) == ' ') ) ptr--;
    // Get substring "variable_type" from var "variable_type"
    token_str = token_str.substr ( ptr_aux, ptr - ptr_aux );
  }
  else {
    // Parameter type
    ptr = token_str.find ( "of" );
    ptr_aux = ptr + 2;
    ptr     = token_str.find_first_of( ":", 0 );
    ptr_aux = token_str.find_first_not_of ( " ", ptr_aux );
    while ( (ptr > 0) && (token_str.at(ptr - 1 ) == ' ') ) ptr--;
    token_str = token_str.substr ( ptr_aux, ptr - ptr_aux );
  }
  // Find the type
  ptr_aux = token_str.find_first_of( " " );
  string tkn_type = token_str.substr( 0, ptr_aux );
  
  /*
   * According to the FlatZinc spec., tkn_type is one of the following:
   * bool, float, int, set, x1..x2.
   */
  if      ( tkn_type.compare ( BOO_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_boolean_domain ();
  }
  else if ( tkn_type.compare ( FLO_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_float_domain ();
  }
  else if ( tkn_type.compare ( INT_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_int_domain ();
  }
  else if ( tkn_type.compare ( SET_TOKEN ) == 0 ) {
    /*
     * According to FlatZinc, set could be on of the following:
     * int, x1..x2, {x1, x2, ..., xk}
     */
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->
    set_subset_domain ( token_str );
  }
  else if ( token_str.find ( RAN_TOKEN ) != std::string::npos ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_range_domain ( token_str );
  }
  else if ( token_str.at ( 0 ) == '{' ) {
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_subset_domain ( token_str );
  }
  else {
    logger->error( _dbg + "Parse Error in variable declaration: " + token_str,
                  __FILE__, __LINE__);
    return nullptr;
  }

  // Read identifier
  token_str.assign( _c_token );
  ptr = token_str.find_first_of( ":", 0 );
  token_str = token_str.substr ( ptr + 1 );
  string var_id = "";
  for ( auto x: token_str ) {
    if ( var_id.size() && x == ' ' ) break;
    if ( x == ' ') continue;
    if ( (x == ';') ||
         (x == ':')) {
      break;
    }
    var_id += x;
  }
  
  // Set array id
  (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_var_id ( var_id );
  
  ptr = token_str.find_first_of( "[", 0 );
  
  if ( ptr != string::npos ) {
    ptr_aux = token_str.find_first_of( "]" );
    
    // Set support array and string representing elements
    string support_element = token_str.substr( ptr, ptr_aux - ptr + 1 );
    (std::static_pointer_cast<TokenArr>( tkn_ptr ))->
    set_support_elements ( support_element );
  }
  
  // Set range array
  token_str.assign( _c_token );
  ptr     = token_str.find_first_of( "[", 0 );
  ptr_aux = token_str.find_first_of( "]", 0 );
  token_str = token_str.substr ( 1, ptr_aux - ptr - 1 );
  
  int lower_bound, upper_bound;
  ptr_aux = token_str.find_first_of( ".", 0 );
  lower_bound = atoi( token_str.substr( 0, ptr_aux ).c_str() );
  ptr_aux = token_str.find_first_of( ":", ptr_aux+2 );
  
  //Check ' ' before ':'
  token_str = token_str.substr( token_str.find_first_of( ".", 0 ) + 2, ptr_aux );
  ptr_aux = token_str.find_first_of( " ", 0 );
  if ( ptr_aux == std::string::npos ) {
    upper_bound = atoi ( token_str.c_str() );
  }
  else {
    upper_bound = atoi ( token_str.substr( 0, ptr_aux ).c_str() );
  }
  
  (std::static_pointer_cast<TokenArr>( tkn_ptr ))->set_array_bounds ( lower_bound, upper_bound );
  
  return tkn_ptr;
}//analyze_token_arr

TokenPtr
FZNTokenization::analyze_token_con () {
  // Token (pointer) to return
  TokenPtr tkn_ptr = make_shared<TokenCon> ();
  
  // Clear line: avoid useless chars
  for ( int i = 0; i < CON_TOKEN.length(); i++ ) _c_token++;
  clear_line();
  
  // Use string methods
  string token_str;
  token_str.assign( _c_token );
  
  // Read constraint identifier
  size_t ptr, ptr_aux;
  ptr     = token_str.find_first_of( "(" );
  ptr_aux = token_str.find_first_of( ")" );
  if ( (ptr     == std::string::npos) ||
       (ptr_aux == std::string::npos) ||
       (ptr_aux < ptr) ) {
    logger->error( _dbg + "Constraint not valid" + _c_token,
                   __FILE__, __LINE__ );
    return nullptr;
  }
  
  (std::static_pointer_cast<TokenCon>( tkn_ptr ))->set_con_id( token_str.substr( 0, ptr ) );
  // Get the expressions that identify the constraint
  token_str = token_str.substr ( ptr + 1, ptr_aux - ptr - 1 );
  
  int brk_counter = 0;
  string expression = "";
  for ( auto x : token_str ) {
    if ( x == '[' ) brk_counter++;
    
    if ( x == ']' ) {
      expression += x;
      if ( brk_counter ) {
        brk_counter--;
        if ( brk_counter == 0 ) {
          (std::static_pointer_cast<TokenCon>( tkn_ptr ))->add_expr ( expression );
          expression.assign("");
        }
      }
    }//']'
    else if ( x == ',' ) {
      if ( brk_counter > 0 ) {
        expression += x;
      }
      else if ( expression.length() ) {
        (std::static_pointer_cast<TokenCon>( tkn_ptr ))->add_expr ( expression );
        expression.assign("");
      }
    }
    else if ( x == ' ' ) {
      if ( brk_counter > 0 ) {
        expression += x;
      }
      else {
        expression.assign("");
      }
    }
    else {
      expression += x;
    }
  }//x
  if ( expression.length() ) {
    (std::static_pointer_cast<TokenCon>( tkn_ptr ))->add_expr ( expression );
  }
  
  return tkn_ptr;
}//analyze_token_con

TokenPtr
FZNTokenization::analyze_token_sol () {
  // Token (pointer) to return
  TokenPtr tkn_ptr = make_shared<TokenSol> ();
  
  // Clear line: avoid useless chars
  for ( int i = 0; i < SOL_TOKEN.length(); i++ ) _c_token++;
  clear_line();
  
  // Use string methods
  string token_str;
  token_str.assign( _c_token );
  
  std::size_t found, found_aux;
  found = token_str.find( SAT_TOKEN );
  if ( found != std::string::npos ) {
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_goal ( SAT_TOKEN );
  }
  
  found = token_str.find( MIN_TOKEN );
  if ( found != std::string::npos ) {
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_goal ( MIN_TOKEN );
    string str_aux = token_str.substr ( found + MIN_TOKEN.size() );
    string var_to_minimize = "";
    for ( auto x : str_aux ) {
      if ( x == ' ' ) continue;
      var_to_minimize += x;
    }
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_var_goal ( var_to_minimize );
  }
  
  found = token_str.find( MAX_TOKEN );
  if ( found != std::string::npos ) {
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_goal ( MAX_TOKEN );
    string str_aux = token_str.substr ( found + MAX_TOKEN.size() );
    string var_to_minimize = "";
    for ( auto x : str_aux ) {
      if ( x == ' ' ) continue;
      var_to_minimize += x;
    }
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_var_goal ( var_to_minimize );
  }
  
  // Check annotations
  found = token_str.find( "::" );
  if ( found != std::string::npos ) {
    found     = token_str.find_first_of ( "(" );
    found_aux = token_str.find_first_of ( ")" );
    if ( (found_aux < found) ||
         (found     == std::string::npos) ||
         (found_aux == std::string::npos) ) {
      logger->error( _dbg + "Parse Error in solve statement: " + token_str,
                    __FILE__, __LINE__);
      return nullptr;
    }
    string strategy = "";
    for ( auto x: token_str ) {
      if ( x == '(' ) break;
      if ( (x == ' ') || (x == ':') ) continue;
      strategy += x;
    }
    (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_params ( strategy );
    
    // Other params within "(" and ")"
    token_str = token_str.substr( found+1, found_aux - found - 1 );
    
    // Check whether there is a set of variables to label
    if ( token_str.at ( 0 ) == '[' ) {
      token_str = token_str.substr( 1 );
      
      bool first_insertion = true;
      int next_position = 0;
      int brk_counter   = 1;
      string expression = "";
      for ( auto x : token_str ) {
        next_position++;
        if ( x == '[' ) {
          brk_counter++;
          if ( brk_counter > 1 && first_insertion ) {
            (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_params ( expression );
            first_insertion = false;
          }
        }
        if ( x == ']' ) {
          expression += x;
          if ( brk_counter > 1 ) {
            brk_counter--;
            if ( brk_counter == 1 ) {
              (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_var_to_label ( expression );
              expression.assign("");
            }
          }
          else if ( brk_counter == 1 ) {
            break;
          }
        }//']'
        else if ( (x == ',') || (x == ' ') ) {
          continue;
        }
        else {
          expression += x;
        }
      }//x

      token_str = token_str.substr( next_position );
      char * pch;
      char * c_str = new char[ token_str.length() + 1 ];
      strncpy ( c_str, token_str.c_str(), token_str.length() );
      pch = strtok ( c_str, " ," );
      while ( pch != NULL ) {
        (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_params ( pch );
        pch = strtok ( NULL, " ," );
      }
      delete [] c_str;
    }
    else {
      char * pch;
      char * c_str = new char[ token_str.length() + 1 ];
      strncpy ( c_str, token_str.c_str(), token_str.length() );
      pch = strtok ( c_str, " ," );
      while ( pch != NULL ) {
        (std::static_pointer_cast<TokenSol>( tkn_ptr ))->set_solve_params ( pch );
        pch = strtok ( NULL, " ," );
      }
      delete [] c_str;
    }
  }
  
  return tkn_ptr;
}//analyze_token_sol

TokenPtr
FZNTokenization::analyze_token_var () {
  
  // Token (pointer) to return
  TokenPtr tkn_ptr = make_shared<TokenVar> ();
  
  // Clear line: avoid useless chars
  for ( int i = 0; i < VAR_TOKEN.length(); i++ ) _c_token++;
  clear_line();
  
  // Use string methods
  string token_str;
  token_str.assign( _c_token );
  
  // Check whether the variable is a support variable
  if ( token_str.find ( VII_TOKEN, 0 ) != string::npos ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_support_var ();
  }
  
  // Read var_type
  size_t ptr;
  ptr = token_str.find_first_of( ":", 0 );
  while ( (ptr > 0) && (token_str.at(ptr - 1 ) == ' ') ) ptr--;
  
  token_str = token_str.substr ( 0, ptr );
  
  /*
   * According to FlatZinc, tkn_type is one of the following:
   * bool, float, int, set, x1..x2.
   */
  if      ( token_str.compare( BOO_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_boolean_domain ();
  }
  else if ( token_str.compare( FLO_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_float_domain ();
  }
  else if ( token_str.compare( INT_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_int_domain ();
  }
  else if ( token_str.compare( SET_TOKEN ) == 0 ) {
    
    /*
     * According to FlatZinc, set could be on of the following:
     * int, x1..x2, {x1, x2, ..., xk}
     */
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->
    set_subset_domain ( token_str );
  }
  else if ( token_str.find ( RAN_TOKEN ) != std::string::npos ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_range_domain ( token_str );
  }
  else if ( token_str.at ( 0 ) == '{' ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_subset_domain ( token_str );
  }
  else {
    logger->error( _dbg + "Parse Error in variable declaration: " + token_str,
                  __FILE__, __LINE__);
    return nullptr;
  }

  // Read identifier
  token_str.assign( _c_token );
  ptr = token_str.find_first_of( ":", 0 );
  token_str = token_str.substr ( ptr + 1 );
  string var_id = "";
  for ( auto x: token_str ) {
    if ( var_id.size() && x == ' ' ) break;
    if ( x == ' ') continue;
    if ( (x == ';') ||
         (x == ':')) {
      break;
    }
    var_id += x;
  }
  
  // Set objective var if found
  if ( var_id.compare( OBJ_TOKEN ) == 0 ) {
    (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_objective_var ();
  }
  
  // Set variable id
  (std::static_pointer_cast<TokenVar>( tkn_ptr ))->set_var_id ( var_id );
  
  return tkn_ptr;
}//analyze_token_var

TokenPtr
FZNTokenization::analyze_token_lns () {
  logger->error( _dbg + "LNS not yet implemented" + _c_token,
                __FILE__, __LINE__ );
  return nullptr;
}//analyze_token_lns





