//
//  depth_first_search.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "depth_first_search.h"

using namespace std;

size_t DepthFirstSearch::_search_id = 0;

DepthFirstSearch::DepthFirstSearch () :
_dbg                 ("DepthFirstSearch - "),
_depth               ( 0 ),
_num_backtracks      ( 0 ),
_num_nodes           ( 0 ),
_num_wrong_decisions ( 0 ),
_backtracks_out      ( -1 ),
_nodes_out           ( -1 ),
_wrong_out           ( -1 ),
_store               ( nullptr ),
_heuristic           ( nullptr ) {
  _search_id++;
}//DepthFirstSearch

DepthFirstSearch::~DepthFirstSearch () {
}//~DepthFirstSearch

void
DepthFirstSearch::set_store ( ConstraintStorePtr store ) {
  _store = store;
}//set_store

void
DepthFirstSearch::set_heuristic ( HeuristicPtr heuristic ) {
  _heuristic = heuristic;
}//set_heuristic

size_t
DepthFirstSearch::get_backtracks () const {
  return _num_backtracks;
}//get_backtracks

size_t
DepthFirstSearch::get_nodes() const {
  return _num_nodes;
}//get_nodes

size_t
DepthFirstSearch::get_wrong_decisions () const {
  return _num_wrong_decisions;
}//get_wrong_decisions

std::vector<DomainPtr>
DepthFirstSearch::get_solution () const {
  throw "Not yet implemented";
}//get_solution

std::vector<DomainPtr>
DepthFirstSearch::get_solution ( int n_sol ) const {
  throw "Not yet implemented";
}//get_solution

void
DepthFirstSearch::set_backtrack_out ( size_t out_b ) {
  _backtracks_out = out_b;
}//set_backtrack_out

void
DepthFirstSearch::set_nodes_out( size_t out_n ) {
  _nodes_out = out_n;
}//set_nodes_out

void
DepthFirstSearch::set_wrong_decisions_out ( size_t out_w ) {
  _wrong_out = out_w;
}//set_wrong_decisions_out

bool
DepthFirstSearch::labeling () {
  
  // Base case: no store implies trivially satisfied.
  if ( _store == nullptr ) return true;
  
  bool search_consistent  = _store->consistency();
  if ( search_consistent ) {
    search_consistent = label( 0 );
  }
  
  // Print info about the search
  print ();
  
  return search_consistent;
}//labeling

bool
DepthFirstSearch::label( int var ) {
  cout << _dbg << "Label V_" << var << endl;
  return true;
}//label

void
DepthFirstSearch::print () const {
  cout << "DepthFirstSearch " << _search_id << ":\n";
  cout << "Number of explored nodes:  " << get_nodes() << endl;
  cout << "Number of backtracks:      " << get_backtracks() << endl;
  cout << "Number of wrong decisions: " << get_wrong_decisions() << endl;
  if ( _heuristic != nullptr ) {
    cout << "Heuristic:\n";
    _heuristic->print ();
  }
}//print


