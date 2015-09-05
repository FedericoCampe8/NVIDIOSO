//
//  simple_constraint_store.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/08/14.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "simple_constraint_store.h"

using namespace std;

SimpleConstraintStore::SimpleConstraintStore () :
_dbg                      ( "SimpleConstraintStore - " ),
_constraint_to_reevaluate ( nullptr ),
_constraint_queue_size    ( 0 ),
_number_of_constraints    ( 0 ),
_number_of_propagations   ( 0 ),
_satisfiability_check     ( true ),
_consistency_propagation  ( true ),
_failure                  ( false ) {
}//SimpleConstraintStore

SimpleConstraintStore::~SimpleConstraintStore () {
  assert( _constraint_to_reevaluate == nullptr );
  _lookup_table.clear     ();
  _constraint_queue.clear ();
}//~SimpleConstraintStore

void
SimpleConstraintStore::fail () 
{
  _failure = true;
}//fail

bool 
SimpleConstraintStore::is_consistency_failed ()
{
	return _failure == true;
}//is_consistency_failed

void
SimpleConstraintStore::handle_failure () {
  _failure = false;
  clear_queue ();
}//handle_failure

void
SimpleConstraintStore::sat_check ( bool sat_check ) {
  _satisfiability_check = sat_check;
}//sat_check

void
SimpleConstraintStore::con_check ( bool con_check )
{
    _consistency_propagation = con_check;
}//con_check

size_t
SimpleConstraintStore::num_constraints () const {
  return _number_of_constraints;
}//num_constraints

size_t
SimpleConstraintStore::num_propagations () const {
  return _number_of_propagations;
}//num_propagations

size_t
SimpleConstraintStore::num_constraints_to_reevaluate () const {
  return _constraint_queue_size;
}//num_constraints
  
void 
SimpleConstraintStore::add_changed ( size_t c_id, EventType event ) {
  /*
   * Check if the constraints belongs to the constraint store.
   * @note it requires log n time, where n = _number_of_constraints.
   */
  if ( _lookup_table.find ( c_id ) == _lookup_table.end() ) return;
  
  /*
   * Check if the constraints is already set for re-evaluation.
   * @note it requires log n time, where n = _constraint_queue_size.
   */ 
  if ( _constraint_queue.insert( c_id ).second ) _constraint_queue_size++;
  
  /*
   * Detach constraint if naive or unary, once propagated one time
   * there is no need to propagate/check it anymore.
   */
   if ( _lookup_table[ c_id ]->is_naive () || _lookup_table[ c_id ]->is_unary () )
   {
   		_lookup_table[ c_id ]->remove_constraint ();
   }
}//add_changed

void
SimpleConstraintStore::add_changed ( vector< size_t >& c_id, EventType event ) {
  if ( !c_id.size () ) return;
  
  for ( size_t c : c_id )
    add_changed ( c, event );
}//add_changed
 
void
SimpleConstraintStore::impose ( ConstraintPtr c ) 
{
	// Sanity check
	if ( c == nullptr ) return;
	
	// Check if the constraints has been already stored
  	if ( _lookup_table.find( c->get_unique_id() ) != _lookup_table.end() ) return;
  
  	// Add c to the constraint store
  	size_t c_id = c->get_unique_id();
  	
  	// Add constraint into the store
  	_lookup_table [ c_id ] = c;
  	_number_of_constraints++;
  	
  	// Add it into the queue for initial propagation
  	add_changed ( c_id, EventType::OTHER_EVT );
}//impose

void
SimpleConstraintStore::clear_queue () 
{
  if ( !_constraint_queue.size () ) return;
  
  // Clear the constraint queue
  _constraint_queue.clear();
  _constraint_queue_size = 0;
}//clear_queue

bool
SimpleConstraintStore::consistency () {
  
  // Check for some failure happened somewhere else
  if ( _failure ) 
  {
	handle_failure ();
    return false;
  }

  /*
   * Loop into the list of constraints to re-evaluate
   * until the fix-point is reached.
   */
  bool succeed = true;
  while ( !_constraint_queue.empty() )
  {    
      _constraint_to_reevaluate = getConstraint ();
      if ( _consistency_propagation )
      {
          _constraint_to_reevaluate->consistency ();
      }
      
      _number_of_propagations++;
		
	  /*
	   * @note consistency failed check is not necessary but it is used here
	   *       to avoid useless satisfiability checks.
	   */
	  succeed &= !is_consistency_failed();
      if ( _satisfiability_check && succeed )
      { 
          succeed &= _constraint_to_reevaluate->satisfied ();
          if ( !succeed ) break;
      }
  }//while
 
  /*
   * @note here it is possible to add the checks for
   *       for nogood learning.
   */
  _constraint_to_reevaluate = nullptr;
  
  if ( !succeed )
  {
      handle_failure ();
      return false;
  }
  
  return true;
}//consistency


Constraint *
SimpleConstraintStore::getConstraint () {
  
  // Get next constraint to re-evalaute
  Constraint * c = _lookup_table[ *_constraint_queue.begin() ].get();
  
  // Erase constraint from the constraint queue
  _constraint_queue.erase( _constraint_queue.begin() );
  _constraint_queue_size--;
  
  return c;
}//getConstraint

void
SimpleConstraintStore::print () const {
  cout << "Constraint Store\n";
  cout << "Attached constraints:      " << _number_of_constraints << endl;
  cout << "Constraints to reevaluate: " << _constraint_queue_size << endl;
  cout << "Number of propagations:    " << _number_of_propagations << endl;
  cout << "Consistency propagation:   ";
  if ( _consistency_propagation ) cout << " Enabled" << endl;
  else                         	  cout << " Disabled" << endl;
  cout << "Satisfiability check:      ";
  if ( _satisfiability_check ) cout << " Enabled" << endl;
  else                         cout << " Disabled" << endl;
  if ( _constraint_to_reevaluate != nullptr ) {
    cout << "Next constraint to re-evaluate:" << endl;
    _constraint_to_reevaluate->print();
  }
}//print


