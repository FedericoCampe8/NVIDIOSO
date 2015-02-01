//
//  simple_backtrack_manager.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 09/08/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "simple_backtrack_manager.h"

SimpleBacktrackManager::SimpleBacktrackManager() :
_dbg           ("SimpleBacktrackManager - "),
_current_level ( 0 ) {
}//SimpleBacktrackManager

SimpleBacktrackManager::~SimpleBacktrackManager () {
  _backtrackable_objects.clear();
  _changed_backtrackables.clear();
}//~SimpleBacktrackManager

void
SimpleBacktrackManager::attach_backtracable ( BacktrackableObject * bkt_obj ) {
  _backtrackable_objects[ bkt_obj->get_backtrackable_id() ] = bkt_obj;
  _changed_backtrackables.insert( bkt_obj->get_backtrackable_id() );
}//attach_backtracable

void
SimpleBacktrackManager::detach_backtracable ( size_t bkt_id ) {
  if ( _backtrackable_objects.find ( bkt_id ) != _backtrackable_objects.end() ) {
    _backtrackable_objects.erase( bkt_id );
  }
}//detach_backtracable

size_t
SimpleBacktrackManager::get_level () const {
  return _current_level;
}//get_level

void
SimpleBacktrackManager::add_changed ( size_t idx ) {
  _changed_backtrackables.insert( idx );
}//notify_changed

void
SimpleBacktrackManager::force_storage () {
  for ( auto backtrackable : _backtrackable_objects )
    add_changed ( backtrackable.first );
}//force_storage

void
SimpleBacktrackManager::set_level ( size_t lvl ) {
  
  // It is possible to add only higher levels
  if ( lvl <= _current_level ) return;
  
  /*
   * If trail stack is empty and the current state
   * of the search has to be recorded, then store every
   * backtrackable since they may have not changed 
   * (e.g., a value has been subtracted from the domain of
   * a variable but the other variables do not have any change)
   * but they need to be stored for later propagation.
   */
  if ( _trail_stack.empty() )
    force_storage ();
  
  std::vector < std::pair< size_t, Memento * > > mementos;
  std::vector < size_t    > changed_ids;
  if ( !_changed_backtrackables.empty() ) {
    
    // Prepare the list of Memento objects
    //std::vector < size_t    > changed_ids;
    //std::vector < std::pair< size_t, Memento * > > mementos;
    for ( auto backtrackable : _changed_backtrackables ) {
      auto back_obj = _backtrackable_objects[ backtrackable ];
      
      back_obj->set_state();
      mementos.push_back( std::make_pair(back_obj->get_backtrackable_id(),
                                         back_obj->create_memento()));
      
      changed_ids.push_back( back_obj->get_backtrackable_id() );
    }
    
    // Add the current search state representation to the trail stack
    _trail_stack.push( std::make_pair( _current_level, mementos ) );
    _trail_stack_info.push( changed_ids );
    
    mementos.clear();
  }
  else {
    _trail_stack.push( std::make_pair( _current_level, mementos ) );
    _trail_stack_info.push( changed_ids );
  }
  
  // Update current level and clear the list of changed backtrackables objects
  _current_level = lvl;
  _changed_backtrackables.clear();
  
}//set_level

void
SimpleBacktrackManager::remove_level ( size_t lvl ) {
  if ( lvl != _current_level ) return;
  
  if ( _trail_stack.size () ) {
    if ( (_trail_stack.top()).second.size () ) {
    for ( auto memento_on_top : (_trail_stack.top()).second ) {
        auto backtrackable = _backtrackable_objects[ memento_on_top.first ];
        backtrackable->set_memento( *memento_on_top.second );
        backtrackable->restore_state ();
        delete memento_on_top.second;
      }
    }
    _current_level = _trail_stack.top().first;
    
    // Remove states from trail stack
    _trail_stack.pop ();
    _trail_stack_info.pop ();
  }
  else {
    _current_level = 0;
  }

}//remove_level

void
SimpleBacktrackManager::remove_until_level ( size_t lvl ) {
  while ( lvl > get_level() ) {
    remove_level( lvl-- );
  }
}//remove_until_level

size_t
SimpleBacktrackManager::number_backtracable () const {
  return _backtrackable_objects.size ();
}//number_backtracable

size_t
SimpleBacktrackManager::number_changed_backtracable () const {
  return _changed_backtrackables.size ();
}//number_backtracable

void
SimpleBacktrackManager::print () const {
  std::cout << "Simple Backtrack Manager:" << std::endl;
  std::cout << "current level:                     " <<
  get_level() << std::endl;
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



