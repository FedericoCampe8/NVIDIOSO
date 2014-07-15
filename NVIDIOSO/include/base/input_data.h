//
//  input_data.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.

//  InputData class:
//  Singleton class used for reading input options given by the user.


#ifndef NVIDIOSO_input_data_h
#define NVIDIOSO_input_data_h

#include "globals.h"

class InputData {
  
private:
  // Debug string
  std::string _dbg;
  //! Static instance for singleton
  static InputData* _id_instance;
  // Other flags and params
  int _verbose;
  int _time;
  int _max_sol;
  std::string _in_file;
  std::string _out_file;
  
  // Private methods
  void init();
  void print_help ();
  void print_gpu_info ();
  
protected:
  
  /**
   * Protected constructor: a client cannot instantiate
   * Singleton directly. 
   */
  InputData ( int argc, char* argv[] );
  
public:
  //! Constructor get (static) instance
  static InputData* get_instance ( int argc, char* argv[] ) {
    if ( _id_instance == nullptr ) {
      _id_instance = new InputData ( argc, argv );
    }
    return _id_instance;
  }//get_instance
  
  // Get functions
  bool verbose () const;
  bool timer () const;
  int  max_n_sol () const;
  
  //! Get input file (path to)
  std::string get_in_file  () const;
  
  //! Get output file (path to)
  std::string get_out_file () const;
};

#endif
