//
//  soft_constraint_store.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//


#include "soft_constraint_store.h"

using namespace std;

SoftConstraintStore::SoftConstraintStore () :
	SimpleConstraintStore () {
	_dbg = "SoftConstraintStore - ";
	_all_soft_constraints     = false;
	_all_hard_constraints     = false;
	_force_soft_consistency   = false; 
	_unsat_constraint_level   = 0;
	_unsat_constraint_counter = 0;
	_bitset_index             = 0;
}//SoftConstraintStore

SoftConstraintStore::~SoftConstraintStore () {
}//~SoftConstraintStore

void 
SoftConstraintStore::reset_failure ()
{
	_failure = false;
}//reset_failure

bool SoftConstraintStore::is_hard ( Constraint* c ) const
{
	return ( _hard_constraint_set.find ( c->get_unique_id() ) != _hard_constraint_set.end () );
}//is_hard

bool SoftConstraintStore::is_soft ( Constraint* c ) const
{
	return ( _soft_constraint_set.find ( c->get_unique_id() ) != _soft_constraint_set.end () );
}//is_soft

void 
SoftConstraintStore::impose_all_soft ()
{
	_all_soft_constraints = true;
}//impose_all_soft

void 
SoftConstraintStore::impose_all_hard ()
{
	_all_hard_constraints = true;
}//impose_all_soft

void 
SoftConstraintStore::reset_unsat_counters ()
{	
	// Init bitset with size equal to the current set of attached constraints
	if ( _sat_constraint_bitset.size () == 0 )
	{
		// Reset unsat constraints counter
		_unsat_constraint_counter = 0;
		
		// At the beginning, each constraint is supposed to be satisfied
		_sat_constraint_bitset.resize ( _lookup_table.size (), true );
	}
	
	// Init constraint level for each constraint
	if ( _sat_constraint_level.size () == 0 )
	{
		// Reset unsat level counter
		_unsat_constraint_level = 0;
		
		// At the beginning, each constraint is supposed to be satisfied (i.e., level 0.0)
		for ( auto& val : _lookup_table )
			_sat_constraint_level [ val.first ] = 0.0;
	}
}//reset_unsat_counters 
 
void 
SoftConstraintStore::impose ( ConstraintPtr c ) 
{
	// Sanity check
	if ( c == nullptr ) return;
	
	// Add c to the constraint store
  	size_t c_id = c->get_unique_id();
  	
	// Check and mark constraint as soft in case
  	if ( c->is_soft () || (_all_soft_constraints && !_all_hard_constraints) )
  	{
  		_soft_constraint_set.insert ( c_id );
  	}
  	else if ( !c->is_soft () || (_all_hard_constraints && !_all_soft_constraints) )
  	{
  		_hard_constraint_set.insert ( c_id );
  	}
	
	// Set unique position in the bitset for the constraint c_id
	if ( _constraint_2_bitset.find ( c_id ) == _constraint_2_bitset.end() )
	{
		_constraint_2_bitset [ c_id ] = _bitset_index++;
	}
	
	SimpleConstraintStore::impose ( c );
}//impose
  
void 
SoftConstraintStore::force_soft_consistency ( bool force_soft )
{
	_force_soft_consistency = force_soft;
}//force_soft_consistency

void 
SoftConstraintStore::record_unsat_constraint ( Constraint* c, bool sat )
{
	size_t c_id = c->get_unique_id();
	if ( _constraint_2_bitset.find ( c_id ) == _constraint_2_bitset.end () )
	{
		throw NvdException ( (_dbg + "record_unsat_constraint: no index in bitset found").c_str() );
	}
	c_id = _constraint_2_bitset[ c_id ];
	
	if ( c_id >= _sat_constraint_bitset.size () )
	{
		throw NvdException ( (_dbg + "record_unsat_constraint: wrong bit index").c_str() );
	}
	
	_sat_constraint_bitset.set ( c_id, sat );
}//record_unsat_constraint

void 
SoftConstraintStore::record_unsat_value ( Constraint* c, bool sat )
{
	size_t c_id = c->get_unique_id();
	if ( _sat_constraint_level.find ( c_id ) == _sat_constraint_level.end () )
	{
		throw NvdException ( (_dbg + "record_unsat_value: no key in hash table found").c_str() );
	}
	
	if ( sat )
	{
		_unsat_constraint_level -= _sat_constraint_level [ c_id ];
		_sat_constraint_level [ c_id ] = 0;
	}
	else
	{
		_unsat_constraint_level -= _sat_constraint_level [ c_id ];
		_sat_constraint_level [ c_id ] = c->unsat_level ();
		_unsat_constraint_level += _sat_constraint_level [ c_id ];
	}
}//record_unsat_value

void 
SoftConstraintStore::initialize_internal_state ()
{
	reset_state ();
}//initialize_internal_state

void 
SoftConstraintStore::reset_state ()
{// Clear maps and reset counters
	_sat_constraint_level.clear  ();
	_sat_constraint_bitset.clear ();
	
	handle_failure		 ();
	reset_unsat_counters ();
}//reset_state

bool
SoftConstraintStore::consistency () 
{	
	/*
   	 * Loop into the list of constraints to re-evaluate
   	 * until the fix-point is reached.
   	 */
  	bool succeed = true;
  	bool hard_constraint = false;
  	while ( !_constraint_queue.empty() )
 	{   
 		// At each iteration reset failure, only hard constraint failures are considered 
 		reset_failure ();
 		
		_constraint_to_reevaluate = getConstraint ();
      	if ( _consistency_propagation )
      	{
        	_constraint_to_reevaluate->consistency ();
      	}
      	hard_constraint = is_hard ( _constraint_to_reevaluate ) && !_force_soft_consistency;
      
      	_number_of_propagations++;
		
	  	/*
	   	 * @note consistency failed check is not necessary but it is used here
	   	 *       to avoid useless satisfiability checks.
	   	 */
	  	if ( hard_constraint && is_consistency_failed () )
	  	{
	  		succeed = false;
	  		break;
	  	}
	  	
      	if ( _satisfiability_check )
      	{ 
        	bool sat = _constraint_to_reevaluate->satisfied ();
        	
        	if ( !sat && hard_constraint )
        	{// Hard constraint not satisfied
        		succeed = false;
	  			break;
        	}
			else if ( !hard_constraint )
			{// Soft constraint not satisfied
				if ( !sat )
				{// Set current constraint as unsatisfied
					record_unsat_value      ( _constraint_to_reevaluate );
					record_unsat_constraint ( _constraint_to_reevaluate );
				}
				else
				{
					record_unsat_value      ( _constraint_to_reevaluate, true );
					record_unsat_constraint ( _constraint_to_reevaluate, true );
				}
			}
      	}
  	}//while

	// Set unsat constraints and unsat level value after complete propagation
	set_unsat_counters (); 
	
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

void
SoftConstraintStore::set_unsat_counters ( bool force_update )
{
	_unsat_constraint_counter = 
	_sat_constraint_bitset.size () - _sat_constraint_bitset.count ();
	
	/*
	 * The following is not needed by default since
	 * the _unsat_constraint_counter is updated at every iteration.
	 */
	 if ( force_update )
	 {
	 	_unsat_constraint_counter = 0.0;
	 	for ( auto& val : _sat_constraint_level )
	 	{
	 		_unsat_constraint_counter += val.second;
	 	}
	 }
}//set_unsat_counters

std::size_t 
SoftConstraintStore::num_soft_constraints () const
{
	return _soft_constraint_set.size();
}//num_soft_constraints

std::size_t 
SoftConstraintStore::num_hard_constraints () const
{
	return _hard_constraint_set.size();
}//num_hard_constraints

std::size_t 
SoftConstraintStore::num_unsat_constraints () const
{
	return _unsat_constraint_counter;
}//num_unsat_constraints
 
double 
SoftConstraintStore::get_unsat_level_constraints () const
{
	return _unsat_constraint_counter;
}//get_unsat_level_constraints

Constraint *
SoftConstraintStore::get_hard_constraint () 
{ 
	// Get next constraint to re-evalaute
  	Constraint * c;
  	for ( auto& c_id : _constraint_queue )
  	{
  		if ( _hard_constraint_set.find ( c_id ) !=  _hard_constraint_set.end() )
  		{
  			c = _lookup_table[ c_id ].get();
  			
  			// Erase constraint from the constraint queue
  			_constraint_queue.erase( c_id );
  			_constraint_queue_size--;
  		}
  	}
  
  return nullptr;
}//getConstraint

Constraint *
SoftConstraintStore::get_soft_constraint () 
{ 
	// Get next constraint to re-evalaute
  	Constraint * c;
  	for ( auto& c_id : _constraint_queue )
  	{
  		if ( _soft_constraint_set.find ( c_id ) !=  _soft_constraint_set.end() )
  		{
  			c = _lookup_table[ c_id ].get();
  			
  			// Erase constraint from the constraint queue
  			_constraint_queue.erase( c_id );
  			_constraint_queue_size--;
  		}
  	}
  
  return nullptr;
}//getConstraint

void
SoftConstraintStore::print () const {
  cout << "Constraint Store\n";
  cout << "Attached constraints:       " << _number_of_constraints << endl;
  cout << "Number of soft constraints: " << num_soft_constraints () << endl;
  cout << "Number of hard constraints: " << num_hard_constraints () << endl;
  cout << "Constraints to reevaluate:  " << _constraint_queue_size << endl;
  cout << "Number of propagations:     " << _number_of_propagations << endl;
  cout << "Satisfiability check:       ";
  if ( _satisfiability_check ) cout << " Enabled" << endl;
  else                         cout << " Disabled" << endl;
  if ( _constraint_to_reevaluate != nullptr ) {
    cout << "Next constraint to re-evaluate:" << endl;
    _constraint_to_reevaluate->print();
  }
}//print



