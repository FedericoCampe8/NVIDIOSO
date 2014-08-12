/*
 *  input_data.cc
 *  NVIDIOSO
 *
 *  Created by Federico Campeotto on 26/06/14.
 *  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
 */

#include <getopt.h>
#include "input_data.h"

using namespace std;

/// Init static variable
InputData* InputData::_id_instance = nullptr;

void
InputData::init () {
  _dbg = "InputData - ";
  _verbose  = 0;
  _time     = 0;
  _max_sol  = 1;
  _timeout  = -1;
  _in_file  = "";
  _out_file = "";
}//init

InputData::InputData ( int argc, char* argv[] ) {
  
  // Init parameters
  init();
  
  // Read input
  while ( true ) {
    static struct option long_options[] =
    {
      // These options set a flag.
      {"verbose", no_argument, &_verbose,      1},
      {"watcher", no_argument, &_time,         1},
      /* These options don't set a flag.
        We distinguish them by their indices. */
      {"help",        no_argument,        0, 'h'}, // Print help message
      {"device",      no_argument,        0, 'd'}, // Print device info
      {"solutions",   required_argument,  0, 'n'}, // Set number of solutions
      {"timeout",     required_argument,  0, 't'}, // Set timeout in seconds
      {"input",       required_argument,  0, 'i'}, // Set input file
      {"output",      required_argument,  0, 'o'}, // Set output file
      {0, 0, 0, 0}
    };
    // getopt_long stores the option index here.
    int option_index = 0;
    
    int c = getopt_long (argc, argv, "hvdwn:t:i:o:",
                         long_options, &option_index);
    
    // Detect the end of the options.
    if ( c == -1 ) break;
    // Switch on the options read from input
    switch ( c ) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if ( long_options[ option_index ].flag != 0 )
          break;
        printf ( "option %s", long_options[ option_index ].name );
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;
        
      case 'h':
        print_help ();
        exit( 0 );
        
      case 'd':
        print_gpu_info ();
        break;
       
      case 'w':
        _time = true;
        break;
        
      case 'v':
        _verbose = 1;
        break;
        
      case 't':
        _timeout = atof ( optarg );
        break;
        
      case 'n':
        _max_sol = atoi( optarg );
        break;
        
      case 'i':
        _in_file = optarg;
        break;
        
      case 'o':
        _out_file = optarg;
        break;
        
      default:
        print_help ();
        break;
    }
  }//while
  
  /// Exit if the user did not set an input file!
  if ( _in_file.compare( "" ) == 0 ) {
    print_help();
    exit( 0 );
  }
  else {
    //Consistency check
    ifstream infile ( _in_file.c_str(), ifstream::in );
    if ( !infile.is_open() ) {
      std::cerr << "Can't open file " << _in_file << std::endl;
      exit ( 0 );
    }
    infile.close ();
  }
  
  if ( _verbose ) {
    puts ( "verbose flag is set" );
    logger->set_verbose ( true );
  }
  
  // Print any remaining command line arguments (not options).
  if ( optind < argc ) {
    printf ("non-option ARGV-elements: ");
    while ( optind < argc )
      printf ("%s ", argv[ optind++ ]);
    putchar ('\n');
  }
  
}//Input_data

bool
InputData::verbose () const {
  return _verbose;
}//verbose

bool
InputData::timer () const {
  return _time;
}//timer

int
InputData::max_n_sol () const {
  return _max_sol;
}//max_sol

double
InputData::timeout () const {
  return _timeout;
}//timeout

std::string
InputData::get_in_file () const {
  return _in_file;
}//get_in_file

std::string
InputData::get_out_file () const {
  return _out_file;
}//get_out_file

void
InputData::print_gpu_info () {
  std::cerr << "@Todo - print_gpu_info\n";
}//print_help

void
InputData::print_help () {
  cout << "Usage: ./nvidioso -i <infile> [options]\n" << endl;
  cout << "          Options           |          Description      \n";
  cout << "=========================== | ===========================\n";
  cout << " -v|--verbose               | - Printf verbose info\n";
  cout << "                            |   during computation.\n";
  cout << " -h|--help                  | - Print this help message.\n";
  cout << " -d|--help                  | - Print device info message.\n";
  cout << " -w|--watcher               | - Turns on the time-watcher.\n";
  cout << " -i|--input      (string)   | - Read input file.\n";
  cout << " -o|--output     (string)   | - Set output file.\n";
  cout << " -n|--solutions  (int)      | - Set number of solutions:\n";
  cout << "                            |   -1 for all solutions,\n";
  cout << "                            |   (default: 1).\n";
  cout << " -t|--timeout    (double)   | - Set a timeout limit for\n";
  cout << "                            |   solving each given model\n";
  cout << "                            |   (default: inf).\n";
  cout << "=========================== | ===========================\n";
  cout << "You may want to try:\n";
  cout << "\t" << "./nvidioso -i test/nqueens.fzn\n";
  cout << "For any questions, feel free to write at: campe8@nmsu.edu.\n";
}//print_gpu_info


