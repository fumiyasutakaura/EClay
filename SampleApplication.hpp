#ifndef APPLICATION_MAINCANVAS_HPP
#define APPLICATION_MAINCANVAS_HPP

#include "EClaySDK/EClay.hpp"

class MainCanvas : public ECCanvas {
public:
    static ECSmtPtr<MainCanvas> Create( const int& width, const int& height ) {
        MainCanvas *ret = new MainCanvas( width, height );
        return ret;
    }
    
private:
    MainCanvas( const int& width, const int& height ) : ECCanvas( width, height ) {}
    // block
    MainCanvas();
    MainCanvas( const MainCanvas& obj );
    MainCanvas& operator=( const MainCanvas& obj );
    
    ECSmtPtr<ECModel> model;
    ECSmtDevPtr<ECDevLight> devLight;

    int fpscounter_skip_flame;
    
    // inherited pure virtual method
    void onInit();
    void onUpdate();
    void onDraw();
    
};

#endif
