//
//  simple_search_memory_manager.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/28/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "simple_search_memory_manager.h"

using namespace std;

SimpleSearchMemoryManager:: SimpleSearchMemoryManager () {
}

SimpleSearchMemoryManager::~SimpleSearchMemoryManager () {
}//~ GreedyNeighborhoodEvaluator

void 
SimpleSearchMemoryManager::record_state ( ObjectiveState& state )
{
	for ( auto& idx : state.neighborhood_index )
	{
		_tenure [ idx ].push_back ( state );
	}
}//record_state

std::vector< ObjectiveState > 
SimpleSearchMemoryManager::get_state_from ( std::size_t idx, std::size_t k )
{
	std::vector< ObjectiveState > states;
	if ( _tenure.find ( idx ) != _tenure.end () )
	{
		std::size_t size_tenure = _tenure [ idx ].size ();
		int state_idx = size_tenure - k;
		if ( state_idx < 0 ) 
			state_idx = 0;
		for ( ; state_idx < size_tenure; ++state_idx )
		{
			states.push_back ( _tenure [ idx ][ state_idx ] );
		}
	}
	return states;
}//get_state_from

std::vector< ObjectiveState > 
SimpleSearchMemoryManager::get_state ( std::size_t idx )
{
	if ( _tenure.find ( idx ) != _tenure.end () )
	{
		return _tenure [ idx ];
	}
	
	std::vector< ObjectiveState > states;
	return states;
}//get_state

std::vector< ObjectiveState > 
SimpleSearchMemoryManager::get_state ( std::vector< std::size_t > indexes )
{
	std::vector< ObjectiveState > states;
	for ( auto& idx : indexes )
	{
		if ( _tenure.find ( idx ) != _tenure.end () )
		{
			for ( auto& val : _tenure [ idx ] )
			{
				states.push_back ( val );
			}
		}
	}
	return states;
}//get_state

void 
SimpleSearchMemoryManager::clear_memory ( std::size_t idx )
{
	if ( _tenure.find ( idx ) != _tenure.end () )
	{
		_tenure [ idx ].clear();
	}
}//clear_memory

void 
SimpleSearchMemoryManager::clear_memory ()
{
	_tenure.clear ();
}//clear_memory

void 
SimpleSearchMemoryManager::print () const 
{
	cout << "SimpleSearchMemoryManager\n";
}//print