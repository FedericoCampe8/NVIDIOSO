//
//  globals.h
//  NVIDIOSO
//
//  Created by Federico Campeotto on 26/06/14.
//  Copyright (c) 2014 ___UDNMSU___. All rights reserved.
//

#ifndef NVIDIOSO_globals_h
#define NVIDIOSO_globals_h

/* Common dependencies */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>
#include <cstddef>
 
/* Input/Output */
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

/* Arithmetic and Assertions */
#include <cassert>
#include <cmath>
#include <limits>

/* STL dependencies */
#include <algorithm>
#include <iterator>
#include <bitset>
#include <random>
#include <map>
#include <unordered_map>
#include <list>
#include <queue>
#include <stack>
#include <set>
#include <unordered_set>
#include <string>
#include <regex>
#include <vector>
#include <utility>
#include <memory>

/* Environment variable */
#include "envs.h"

/* Global classes */
#include "logger.h"
#include "statistics.h"
#include "id_generator.h"
#include "nvd_exception.h"
#include "param_data.h"
#include "random_generator.h"

/* Cuda */
#if CUDAON
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#if GCC4
#define nullptr NULL
#define noexcept throw()
#endif

#endif
