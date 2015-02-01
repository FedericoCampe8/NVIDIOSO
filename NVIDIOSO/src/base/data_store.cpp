//
//  data_store.cpp
//  NVIDIOSO
//
//  Created by Federico Campeotto on 27/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#include "data_store.h"

DataStore::DataStore( std::string in_file ) {
  _in_file = in_file;
}//DataStore

DataStore::~DataStore() {
}//~DataStore

CPModel *
DataStore::get_model () {
  return _cp_model;
}//get_model

void
DataStore::print_model_variable_info () {
  
}//print_model_variable_info

void
DataStore::print_model_domain_info () {
  
}//print_model_domain_info

void
DataStore::print_model_constraint_info () {
  
}//print_model_constraint_info