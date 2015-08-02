//
//  cuda_constraint_marco.h
//  iNVIDIOSO
//
//  Created by Federico Campeotto on 07/09/15.
//  Copyright (c) 2014-2015 Federico Campeotto. All rights reserved.
//
//  Macros used by cuda_constraint(s)
//

#ifndef NVIDIOSO_cuda_constraint_macro_h
#define NVIDIOSO_cuda_constraint_macro_h

#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

// MAX SIZE BIT MAP
#include "envs.h"

// GLOBAL STATUS/ WORKING STATUS ON DEVICE
#define D_STATUS _working_status

// EVENTS ON DOMAINS
#define NOP_EVT 0
#define SNG_EVT 1
#define BND_EVT 2
#define MIN_EVT 3
#define MAX_EVT 4
#define CHG_EVT 5
#define FAL_EVT 6

// STANDARD REPRESENTATION
#define BIT_REP 0
#define BND_REP 1

// BOOLEAN DOMAIN REPRESENTATION
#define BOL_EVT     7
#define BOL_SNG_EVT 8
#define BOL_F   0
#define BOL_T   1
#define BOL_U   2

// INDEX ON DOMAINS
#define EVT 0
#define REP 1
#define LB  2
#define UB  3
#define DSZ 4
#define BIT 5

// INDEX FOR BOOLEAN STATUS REPRESENTATION
#define ST  1

// VARIABLES (DOMAINS) TYPE
#define MIXED_DOM    1
#define BOOLEAN_DOM  2
#define STANDARD_DOM (5 + (VECTOR_MAX/(8*sizeof(int))))

// STATUS
#define STATUS(x, y) (D_STATUS[(x)][(y)])
#define GET_VAR_EVT(x)D_STATUS[(x)][EVT]
#define GET_VAR_REP(x)D_STATUS[(x)][REP]
#define GET_VAR_LB(x)D_STATUS[(x)][LB]
#define GET_VAR_UB(x)D_STATUS[(x)][UB]
#define GET_VAR_DSZ(x)D_STATUS[(x)][DSZ]

// DEVELOPER "FRIENDLY" ALIAS
#define NUM_ARGS _args_size
#define NUM_VARS _scope_size
#define ARGS     _args
#define VARS     _vars
#define C_ARG    _args[0]
#define LAST_VAR_IDX ((_scope_size > 0) ? (_scope_size - 1) : 0)
#define LAST_ARG_IDX ((_args_size > 0) ? (_args_size - 1) : 0)
#define X_VAR    0
#define Y_VAR    1
#define Z_VAR    2

// SIZE UTILITIES
constexpr int BIT_IDX = 5;
constexpr int BITS_IN_CHUNK = sizeof(int) * 8;
constexpr int NUM_CHUNKS    = VECTOR_MAX / BITS_IN_CHUNK;

using uint = unsigned int;

#endif

