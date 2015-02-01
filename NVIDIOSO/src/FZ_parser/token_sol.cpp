//
//  token_sol.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 07/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "token_sol.h"

using namespace std;

TokenSol::TokenSol () :
Token              ( TokenType::FD_SOLVE ),
_var_goal          ( "" ),
_solve_goal        ( "" ),
_search_choice     ( "" ),
_label_choice      ( "" ),
_variable_choice   ( "" ),
_assignment_choice ( "" ),
_strategy_choice   ( "" ) {
  _dbg = "TokenSol - ";
}//TokenSol

void
TokenSol::set_var_goal ( string var_goal ) {
  if ( !_var_goal.compare ( "" ) ) {
    _var_goal = var_goal;
  }
}//set_var_goal

void
TokenSol::set_solve_goal ( string solve ) {
  if ( !_solve_goal.compare ( "" ) ) {
    _solve_goal.assign( solve );
  }
}//set_solve_goal

void
TokenSol::set_solve_params( string choice ) {
  if ( !_search_choice.compare ( "" ) ) {
    set_search_choice ( choice );
    return;
  }
  if ( !_label_choice.compare ( "" ) ) {
    set_label_choice ( choice );
    return;
  }
  if ( !_variable_choice.compare ( "" ) ) {
    set_variable_choice ( choice );
    return;
  }
  if ( !_assignment_choice.compare ( "" ) ) {
    set_assignment_choice ( choice );
    return;
  }
  if ( !_strategy_choice.compare( "" ) ) {
    set_strategy_choice( choice );
  }
}//set_solve_goal

void
TokenSol::set_search_choice ( string search_choice ) {
  if ( !_search_choice.compare ( "" ) ) {
    _search_choice.assign( search_choice );
  }
}//set_search_choice

void
TokenSol::set_label_choice ( string label_choice ) {
  if ( !_label_choice.compare ( "" ) ) {
    _label_choice.assign( label_choice );
  }
}//set_search_choice

void
TokenSol::set_variable_choice ( string var_choice ) {
  if ( !_variable_choice.compare ( "" ) ) {
    _variable_choice.assign( var_choice );
  }
}//set_search_choice

void
TokenSol::set_assignment_choice ( string assignment_choice ) {
  if ( !_assignment_choice.compare ( "" ) ) {
    _assignment_choice.assign( assignment_choice );
  }
}//set_assignment_choice

void
TokenSol::set_strategy_choice ( string strategy_choice ) {
  if ( !_strategy_choice.compare ( "" ) ) {
    _strategy_choice.assign( strategy_choice );
  }
}//set_strategy_choice

void
TokenSol::set_var_to_label ( string var_to_label ) {
  _var_to_label.push_back( var_to_label );
}//set_var_to_label

string
TokenSol::get_var_goal () const {
  return _var_goal;
}//get_solve_goal

string
TokenSol::get_solve_goal () const {
  return _solve_goal;
}//get_solve_goal

string
TokenSol::get_search_choice () const {
  return _search_choice;
}//get_search_choice

std::string
TokenSol::get_label_choice () const {
  return _label_choice;
}//get_label_choice

string
TokenSol::get_variable_choice () const {
  return _variable_choice;
}//get_variable_choice

string
TokenSol::get_assignment_choice () const {
  return _assignment_choice;
}//get_assignment_choice

string
TokenSol::get_strategy_choice () const {
  return _strategy_choice;
}//get_strategy_choice

int
TokenSol::num_var_to_label () const {
  return (int) _var_to_label.size();
}//get_num_var_to_label

const vector< std::string >
TokenSol::get_var_to_label () const {
  return _var_to_label;
}//get_var_to_label

string
TokenSol::get_var_to_label ( int idx ) const {
  if ( (idx < 0) ||(idx >= _var_to_label.size()) ) {
    return "";
  }
  return _var_to_label [ idx ];
}//get_var_to_label

void
TokenSol::print () const {
  cout << "Solve:\n";
  cout << "Goal - " << _solve_goal << " ";
  if ( _var_goal.compare ( "" ) ) {
    cout << " on " << _var_goal;
  }
  cout << "\n";
  if ( _search_choice.compare ( "" ) ) {
    cout << "Search - " << _search_choice << "\n";
  }
  if ( _label_choice.compare ( "" ) ) {
    cout << "Label Var - " << _label_choice << "\n";
  }
  if ( num_var_to_label() ) {
    cout << "On:\n";
    for ( auto x : _var_to_label ) {
      cout << x << " ";
    }
    cout << "\n";
  }
  if ( _variable_choice.compare ( "" ) ) {
    cout << "Variable choice - " << _variable_choice << "\n";
  }
  if ( _assignment_choice.compare ( "" ) ) {
    cout << "Assignment choice - " << _assignment_choice << "\n";
  }
  if ( _strategy_choice.compare ( "" ) ) {
    cout << "Strategy choice - " << _strategy_choice << "\n";
  }
}//print



