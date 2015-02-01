#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

class Shape;
__device__ Shape* dataptr [ 2 ];

class Shape {
public:
  __device__ Shape (){}
  __device__ virtual ~Shape() {}
  __device__ virtual void draw() const {
    printf("Shape\n");
  }
};

class Point : public Shape {
private:
  int _x, _y;
public:
  __device__ Point ( int x, int y ) :
  _x(x),
  _y(y) {}
  
  __device__ void draw () const {
    printf("Point : public Shape %d %d\n", _x, _y);
  }
};


__global__ void allocMemory () {
  Shape* shape_a = new Shape ();
  Shape* shape_b = new Point (2, 3);
  dataptr [ 0 ] = shape_a;
  dataptr [ 1 ] = shape_b;
}//allocMemory

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
