#include <cutil.h>
#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "ECPixelBuffer.hpp"





__global__ void clearPixels( ECPixel* pixels, const float4 color, float4* zBuffer, const float4 z ) {
    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int offset = x + y * blockDim.x * gridDim.x;
    const float4 point = make_float4(x,y,1.0f,1.0f);
    
    pixels[offset].r = color.x;
    pixels[offset].g = color.y;
    pixels[offset].b = color.z;
    pixels[offset].a = color.w;
    
    zBuffer[offset].x = z.x;
    zBuffer[offset].y = z.y;
    zBuffer[offset].z = z.z;
    zBuffer[offset].w = z.w;
    
}
extern "C" void gpuClearPixels( ECPixel* pixels, const int& width, const int& height, const float4& color, float4* zBuffer, const float4& z ) {
    
    dim3 screen_grid(width/16, height/16);
    dim3 screen_block(16, 16);
    clearPixels<<<screen_grid,screen_block>>>( pixels, color, zBuffer, z );
    
}

    


