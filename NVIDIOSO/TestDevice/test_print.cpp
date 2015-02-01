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


#include <cuda.h>
#include <cuda_runtime_api.h>

__global__ void kernel () {
	   printf("ciao\n");
  //dataptr [ blockIdx.x ]->draw();
  //delete dataptr [ blockIdx.x ];
}//derive

int main ( int argc, char* argv[] ) {
  //allocMemory<<<1,1>>>();
  kernel<<<2,1>>>();
  cudaDeviceSynchronize();
  cudaDeviceReset();
  return 0;
}
