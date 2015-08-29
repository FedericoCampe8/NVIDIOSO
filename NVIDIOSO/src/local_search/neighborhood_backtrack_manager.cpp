//
//  neighborhood_backtrack_manager.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/26/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "neighborhood_backtrack_manager.h"

NeighborhoodBacktrackManager::NeighborhoodBacktrackManager() :
	SimpleBacktrackManager() {
	_dbg = "NeighborhoodBacktrackManager - ";
	_sequence_id = 0;
}//NeighborhoodBacktrackManager

NeighborhoodBacktrackManager::~NeighborhoodBacktrackManager () {
}//~NeighborhoodBacktrackManager

void
NeighborhoodBacktrackManager::reset_sequence_id ()
{
	_sequence_id = 0;
}//reset_sequence_id

void
NeighborhoodBacktrackManager::attach_backtracable ( BacktrackableObject * bkt_obj ) 
{
	_sequence_backtrackable_objects[ _sequence_id++ ] = bkt_obj->get_backtrackable_id();
	SimpleBacktrackManager::attach_backtracable ( bkt_obj );
}//attach_backtracable

void
NeighborhoodBacktrackManager::detach_backtracable ( size_t bkt_id ) 
{
	bool found = false;
	std::size_t key {};
	for ( auto& obj : _sequence_backtrackable_objects )
	{
		if ( obj.second == 	bkt_id )
		{	
			key = obj.first;
			found = true;
			break;
		}
	}
	
	if ( found )
	{
		_sequence_backtrackable_objects.erase ( key );
	}
	SimpleBacktrackManager::detach_backtracable ( bkt_id );
}//detach_backtracable
 
void
NeighborhoodBacktrackManager::set_level ( size_t lvl ) 
{
	// Always store every state at every level
	force_storage ();
		 
	SimpleBacktrackManager::set_level ( lvl );

	// Set flat trail stack if not yet set
	if ( _flat_trail_stack.size () == 0 )
	{
		flatten_stack ();
	}
}//set_level

void
NeighborhoodBacktrackManager::flatten_stack () 
{
	// Reset stack if not empty
	if ( _flat_trail_stack.size() > 0 )
	{
		for ( auto& obj : _flat_trail_stack )
		{
			delete obj.second;
		}
		_flat_trail_stack.clear();
	}
	
	// Set the storage on all backtrackables objects
	force_storage ();

  	if ( !_changed_backtrackables.empty() ) 
  	{
    	for ( auto backtrackable : _changed_backtrackables ) 
    	{
      		auto back_obj = _backtrackable_objects[ backtrackable ];
      		back_obj->set_state();
      		_flat_trail_stack [ back_obj->get_backtrackable_id() ] = back_obj->create_memento();
    	}
  	}
  
  	// Clear the list of changed backtrackables objects previously set by force_storage()
  	_changed_backtrackables.clear();
}//flatten_stack

void
NeighborhoodBacktrackManager::remove_level_on_var ( std::size_t id ) 
{
	if ( _sequence_backtrackable_objects.find ( id ) != _sequence_backtrackable_objects.end () )
  	{
  		std::size_t obj_id = _sequence_backtrackable_objects [ id ];
  		if ( _backtrackable_objects.find ( obj_id ) == _backtrackable_objects.end () )
  		{
  			throw NvdException ( (_dbg + "remove_level_on_var: something happened, no id found in _backtrackable_objects.").c_str() );
  		}
  		
		auto backtrackable = _backtrackable_objects[ obj_id ];
		if ( _flat_trail_stack.find ( obj_id ) == _flat_trail_stack.end() )
		{
			throw NvdException ( (_dbg + "remove_level_on_var: something happened, no id found in _flat_trail_stack.").c_str() );
		}
		backtrackable->set_memento( *_flat_trail_stack [ obj_id ] );
		backtrackable->restore_state ();
  	}
}//remove_level

void
NeighborhoodBacktrackManager::print () const 
{
  std::cout << "Neighborhood Backtrack Manager:" << std::endl;
  std::cout << "current level:                     " <<
  get_level() << std::endl;
  std::cout << "sequence id:                       " <<
  _sequence_id << std::endl;
  std::cout << "Trail stack's size:                " <<
  _trail_stack.size() << std::endl;
  if ( _trail_stack.size() ) {
    std::cout << "Top - ";
    std::cout << "Level: " << _trail_stack.top().first << " Size: " <<
    _trail_stack.top().second.size() << std::endl;
  }
  std::cout << "Number of attached backtrackables: " <<
  number_backtracable() << std::endl;
  std::cout << std::endl;
}//print



