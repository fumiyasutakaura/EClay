#include "ECPixelBuffer.hpp"
#include "../smart_pointer/ECSmtPtr.hpp"
#include <stdlib.h>

#include <cutil_math.h>
#include <cutil_inline.h>

#include "../etc/ECUtil.hpp"
#include "../math/ECQuaternion.hpp"
#include "../math/ECVector.hpp"
#include "../application/ECModel.hpp"

using namespace std;

ECSmtPtr<ECPixelBuffer> ECPixelBuffer::Create( int w, int h ) {
    ECPixelBuffer *ret = new ECPixelBuffer();
    ret->init( w, h );
    return ret;
}

void ECPixelBuffer::init( int w, int h ) {
    
    width = w;
    height = h;
    
    ECPixel* newPtr = (ECPixel*)malloc( sizeof(ECPixel)*w*h );
    pixels = newPtr;
    for(int i=0; i<w*h; i+=1) {
        pixels[i].r = 1.0;
        pixels[i].g = 1.0;
        pixels[i].b = 1.0;
        pixels[i].a = 1.0;
    }
    
    const int size = w * h;
    ECPixel *f = (ECPixel*)malloc( sizeof(ECPixel) * size );
    for( int i=0; i<size; ++i ) {
        f[i].r = 0.0;
        f[i].g = 0.0;
        f[i].b = 0.0;
        f[i].a = 1.0;
    }
    devPixels = ECSmtDevPtr<ECPixel>( f, sizeof(ECPixel) * size );
    free( f );
    
    float4 *hostZBuffer = (float4*)malloc( sizeof(float4)*width*height );
    zBuffer = ECSmtDevPtr<float4>(hostZBuffer, sizeof(float4)*width*height);
    free( hostZBuffer );
}

void ECPixelBuffer::update() {

}

extern "C" void gpuClearPixels( ECPixel* pixels, const int& width, const int& height, const float4& color,
                     float4* zBuffer, const float4& z );
void ECPixelBuffer::clear() {
    clear( 0.0f, 0.0f, 0.0f );
}
void ECPixelBuffer::clear( float r, float g, float b ) {
    const float far = -FLOAT_MAX;
    gpuClearPixels( devPixels.getPtr(), width, height, make_float4(0.0f,0.0f,0.0f,1.0f), zBuffer.getPtr(), make_float4(far,far,far,far) );
}



void ECPixelBuffer::draw() {
    
    devPixels.copyDeviceToHost( pixels.getPtr() );
}
