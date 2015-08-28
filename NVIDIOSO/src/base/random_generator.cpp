//
//  random_generator.cpp
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//

#include "random_generator.h"

using namespace std;

//Global instance
RandomGenerator& glb_rand = RandomGenerator::get_instance ();
 
RandomGenerator::RandomGenerator () :
	_generator    ( std::chrono::system_clock::now().time_since_epoch().count() ),
	_uniform_lb   ( 0 ),
	_uniform_ub   ( std::numeric_limits<int>::max() - 1 ),
	_normal_mean  ( 0.0 ),
	_normal_sigma ( 1.0 ),
	_poisson_mean ( 1.0 ) {
}//RandomGenerator

RandomGenerator::~RandomGenerator () {
}//~RandomGenerator

int 
RandomGenerator::uniform_rand ( int b )
{
	return uniform_rand_param ( 0, b - 1 );
}//uniform_rand

int
RandomGenerator::uniform_rand_param ( int a, int b )
{
	// Sanity check
	if ( b < a ) b = a;
	return (a + (_uniform_dis( _generator ) % (b - a + 1)));
}//uniform_rand

double 
RandomGenerator::normal_rand ( double mean )
{
	return normal_rand_param ( mean, 1.0 );
}//normal_rand

double 
RandomGenerator::normal_rand_param ( double mean, double sd )
{
	if ( mean != _normal_mean || sd != _normal_sigma )
	{
		_normal_mean  = mean;
		_normal_sigma = sd;
	}
	return _normal_mean + _normal_sigma * _normal_dis ( _generator );
}//normal_rand

int 
RandomGenerator::poisson_rand ()
{
	return _normal_dis ( _generator );
}//normal_rand