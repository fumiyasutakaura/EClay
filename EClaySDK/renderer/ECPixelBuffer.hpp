#ifndef EC_PIXELBUFFER_HPP
#define EC_PIXELBUFFER_HPP

#include "../smart_pointer/ECSmtPtr.hpp"

#include <vector_types.h>

struct ECPixel{
    float r,g,b,a;
};

class ECPixelBuffer {
public:
    static ECSmtPtr<ECPixelBuffer> Create( int w, int h );
    void clear();
    void clear( float r, float g, float b );
    void update();
    void draw();
    ECSmtPtr<ECPixel> getPixels() {
        return pixels;
    };
    int getWidth() { return width; }
    int getHeight() { return height; }
    ECSmtDevPtr<ECPixel> getDevPixels() { return devPixels; };
    ECSmtDevPtr<float4> getZBuffer() { return zBuffer; };
    
private:
    ECSmtDevPtr<ECPixel> devPixels;
    ECSmtPtr<ECPixel> pixels;
    int width;
    int height;
    
    ECSmtDevPtr<float4> zBuffer;

    void init( int w, int h );
};

#endif
