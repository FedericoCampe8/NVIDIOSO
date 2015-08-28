//
//  random_generator.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 08/27/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Singleton class to generate random numbers with different probability distributions
//  in different ranges.
//  We declare a global singleton generator object here. 
//  This is important because we don't want to create a new pseudo-random number
// 	generator at every call.
//

#ifndef NVIDIOSO_random_generator_h
#define NVIDIOSO_random_generator_h

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <random>

#if GCC4
#define nullptr NULL
#define noexcept throw()
#endif


class RandomGenerator;
extern RandomGenerator& glb_rand;

class RandomGenerator {
protected:

	//! Random generator engine from a time-based seed
  	std::default_random_engine _generator;
    
    //! Uniform distribution
    std::uniform_int_distribution<int> _uniform_dis;
    
    int _uniform_lb;
    int _uniform_ub;
    
    //! Normal distribution
	std::normal_distribution<double> _normal_dis;
	
	double _normal_mean;
	double _normal_sigma;
	
	//! Posson distribution
	std::poisson_distribution<int> _poisson_dis;
	
	double _poisson_mean;
	
  	RandomGenerator ();
  	
public:

	virtual ~RandomGenerator ();
	
    RandomGenerator ( const RandomGenerator& other )            = delete;
    RandomGenerator& operator= ( const RandomGenerator& other ) = delete;

    //! Get (static) instance (singleton) of Statistics
    static RandomGenerator& get_instance () {
        static RandomGenerator random_generator;
        return random_generator;
    }//get_instance
    
    /**
     * Return uniform random number in [0, b-1]
     * @param b upper bound range distribution (excluded)
     * @return random number using uniform distribution.
     */
    virtual int uniform_rand ( int b );
    
    /**
     * Return uniform random number in [a , b].
     * @param a lower bound range distribution (included)
     * @param b upper bound range distribution (included)
     * @return random number using uniform distribution.
     */
    virtual int uniform_rand_param ( int a = 0, int b = std::numeric_limits<int>::max() - 1 );
    
    /**
     * Return normal random number with mean a and sd b
     * @param a mean of the normal distribution (default 0)
     * @return random number using normal distribution with mean a and sd 1.
     */
    virtual double normal_rand ( double mean = 0 );
    
    /**
     * Return normal random number with mean a and sd b
     * @param a mean of the normal distribution (default 0)
     * @param b standard deviation of the normal distribution (default 1)
     * @return random number using normal distribution.
     */
    virtual double normal_rand_param ( double mean = 0, double sd = 1.0 );
    
    /**
     * Return poisson random number with mean 1.0
     * @return random number using poisson distribution with mean mean.
     */
    virtual int poisson_rand ();
};

#endif
