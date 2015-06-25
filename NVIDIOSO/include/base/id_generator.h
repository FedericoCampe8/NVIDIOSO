//
//  id_generator.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 03/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  Singleton class to generate uniques ids for all the objects
//  defined in the whole program.

#ifndef NVIDIOSO_id_generator_h
#define NVIDIOSO_id_generator_h

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#if GCC4
#define nullptr NULL
#define noexcept throw()
#endif


class IdGenerator;

//! IdGenerator instance: global to all classes if included
extern IdGenerator* glb_id_gen;

class IdGenerator {
  
private:
  //! Static instance for singleton
  static IdGenerator* _id_gen_instance;
  //! Integer Id
  int _int_id;
  int _str_id;
  
  //! String id
  std::string _str_prefix;
  
protected:
  
  /**
   * Protected constructor: a client cannot instantiate
   * Singleton directly.
   */
  IdGenerator ();
  
  //! Convert numbers to string
  std::string n_to_str ( int );
  
public:
  
  //! Constructor get (static) instance
  static IdGenerator* get_instance () {
    if ( _id_gen_instance == nullptr ) {
      _id_gen_instance = new IdGenerator ();
    }
    return _id_gen_instance;
  }//get_instance
  
  //! Reset id generator.
  void reset_int_id ();
  
  //! Reset id generator.
  void reset_str_id ();
  
  //! Set (base) ids (if not already set).
  void set_base_offset ( int );
  
  //! Set (base) ids (if not already set)
  void set_base_prefix ( std::string );
  
  //! Get a new unique int id.
  int         get_int_id ();
  
  //! Get a new unique string id.
  std::string get_str_id ();
  
  //! Get a new unique int id.
  int         new_int_id ();
  
  //! Get a new unique string id.
  std::string new_str_id ();
  
  //! Get the current id already generated.
  int         curr_int_id ();
  
  //! Get the current id already generated.
  std::string curr_str_id ();
  
  // Print info
  void print_int_id ();
  void print_str_id ();
};

#endif
