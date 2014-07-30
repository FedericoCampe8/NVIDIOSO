//
//  domain_test.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 24/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "globals.h"
#include "cuda_variable.h"
#include "int_ne.h"

using namespace std;

int main ( int argc, char * argv[] ) {
  std::string dbg = "domain_test - ";
  
  // Defining two IntVariables
  IntVariablePtr x_var = make_shared<CudaVariable>( 0 );
  IntVariablePtr y_var = make_shared<CudaVariable>( 1 );
  
  // Set vars names
  x_var->set_str_id ( "x" );
  y_var->set_str_id ( "y" );
  
  // Set their domains
  x_var->set_domain ( 5, 5 );
  y_var->set_domain ( 5, 10 );
  
  // Print vars
  x_var->print();
  y_var->print();
  
  // Defining two int values
  int x_int = 5;
  int y_int = 6;
  
  vector<VariablePtr> var_vec = { x_var, y_var };
  
  // Defining int_ne constraints
  IntNe c_neq_a( x_var, y_var );
  c_neq_a.print();
  c_neq_a.print_semantic();
  
  IntNe c_neq_b( x_var, y_int );
  c_neq_b.print();
  c_neq_b.print_semantic();
  
  IntNe c_neq_c( x_int, y_int );
  c_neq_c.print();
  c_neq_c.print_semantic();
  
  // Perform consistency
  c_neq_a.consistency ();
  c_neq_b.consistency ();
  c_neq_c.consistency ();
  
  // Check if constraints are satisfied
  cout << "First constraint:\t";
  if ( c_neq_a.satisfied() ) cout << "SATISFIED\n";
  else                       cout << "UNSATISFIED\n";
  
  cout << "Second constraint:\t";
  if ( c_neq_b.satisfied() ) cout << "SATISFIED\n";
  else                       cout << "UNSATISFIED\n";
  
  cout << "Third constraint:\t";
  if ( c_neq_c.satisfied() ) cout << "SATISFIED\n";
  else                       cout << "UNSATISFIED\n";
  
  // Print vars
  cout << endl;
  cout << "Variables after consistency:\n";
  x_var->print();
  y_var->print();
  
  
  
  cout << dbg << "Exit from int_ne test.\n";
  
  return 0;
}//main
