#ifndef EC_RENDERSTATE_CUH
#define EC_RENDERSTATE_CUH

#include "../smart_pointer/ECSmtPtr.hpp"



class ECRenderState {
public:
    static ECSmtPtr<ECRenderState> Create() {
        ECRenderState *ret = new ECRenderState();
        ret->init();
        return ret;
    }
    int getScreenWidth() { return screenWidth; }
    void setScreenWidth( const int& w ) { screenWidth = w; }
    int getScreenHeight() { return screenHeight; }
    void setScreenHeight( const int& h ) { screenHeight = h; }
    
    float getNear() { return near; }
    void setNear( const float& n ) { near = n; }
    float getFar() { return far; }
    void setFar( const float& f ) { far = f; }
    float getAngleOfView_Degree() { return angleOfView_Degree; }
    void setAngleOfView_Degree( const float& degree ) { angleOfView_Degree = degree; }
    
    ECMatrix getProjectionMatrix() {
        return ECMatrix::Projection( screenWidth, screenHeight, angleOfView_Degree, near, far );
    }
    ECMatrix getScreenMatrix() {
        return ECMatrix::Screen( screenWidth, screenHeight, near, far );
    }
    ECMatrix getViewMatrix() {
        // TODO: eye dir up
        return viewMatrix;
    }
    
private:
    ECRenderState(){}
    ECRenderState(const ECRenderState& rhs);
    ECRenderState& operator=(const ECRenderState& rhs);
    
    void init() {
        projectionMatrix = ECMatrix();
        screenMatrix = ECMatrix();
        viewMatrix = ECMatrix();
    }
    
    int screenWidth;
    int screenHeight;
    
    float near;
    float far;
    float angleOfView_Degree;
    
    ECMatrix projectionMatrix;
    ECMatrix screenMatrix;
    ECMatrix viewMatrix;
};



#endif
