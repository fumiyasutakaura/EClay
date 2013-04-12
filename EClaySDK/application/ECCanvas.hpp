#ifndef EC_CANVAS_HPP
#define EC_CANVAS_HPP

#include "../renderer/ECPixelBuffer.hpp"
#include "ECModel.hpp"
#include "../math/ECVector.hpp"
#include "../math/ECMatrix.hpp"
#include "../math/ECQuaternion.hpp"
#include "../etc/ECUtil.hpp"

#include <cutil_math.h>
#include <cutil_inline.h>

class ECCanvas {
public:
    virtual void onInit() = 0;
    virtual void onUpdate() = 0;
    virtual void onDraw() = 0;
    
    ECSmtPtr<ECPixelBuffer> getPixelBuffer() { return pixelBuffer; }
    
    void update();
    void draw();
    
protected:
    // constructor
    explicit ECCanvas( const int& width, const int& height );
    
    ECSmtPtr<ECPixelBuffer> pixelBuffer;
    ECSmtPtr<ECRenderState> renderState;
    
    void render( ECSmtPtr<ECModel> model, ECSmtDevPtr<ECDevLight> devLight );
    
private:
    // > block --------
    ECCanvas();
    ECCanvas( const ECCanvas& obj );
    ECCanvas& operator=( const ECCanvas& obj );
    // < block --------
    
    bool firstFlag;
    void checkFlagAndInit();
    
};



#endif
