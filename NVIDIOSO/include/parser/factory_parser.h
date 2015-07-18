//
//  factory_parser.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 08/07/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//
//  It decouples the instantiation of a parser, allowing the client more
//  freedom when implementing a new store and/or a parser.
//

#ifndef NVIDIOSO_factory_parser_h
#define NVIDIOSO_factory_parser_h

#include "globals.h"
#include "parser.h"
#include "fzn_parser.h"

class FactoryParser {
public:
  //! Get the right parser based on the input
  static Parser* get_parser ( ParserType pt ) {
    switch ( pt ) {
      case ParserType::FLATZINC:
        return new FZNParser();
        
      default:
        return nullptr;
    }
  }//get_parser
};


#endif
