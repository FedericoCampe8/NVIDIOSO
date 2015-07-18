//
//  factory_cp_model.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/16/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  It decouples the instantiation of a parser, allowing the client more
//  freedom when implementing a new store and/or a parser.
//

#ifndef NVIDIOSO_factory_cp_model_h
#define NVIDIOSO_factory_cp_model_h

#include "globals.h"
#include "cuda_cp_model.h"
#include "cuda_cp_model_simple.h"

enum class CPModelType
{
    CP_MODEL,
        CUDA_CP_MODEL,
        CUDA_CP_MODEL_SIMPLE,
        OTHER
};

class FactoryCPModel {
public:
    
    //! Get the right parser based on the input
    static CPModel* get_cp_model ( CPModelType cp_model )
        {
            switch ( cp_model )
            {
                case CPModelType::CP_MODEL:
                    return new CPModel ();
                case CPModelType::CUDA_CP_MODEL:
                    return new CudaCPModel ();
                case CPModelType::CUDA_CP_MODEL_SIMPLE:
                    return new CudaCPModelSimple ();
                default:
                    return nullptr;
            }
        }//get_parser
};

#endif
