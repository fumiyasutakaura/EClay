//
//  ECCanvas.cpp
//  EClay001
//
//  Created by Fumiyasu Takaura on 3/12/13.
//
//

#include "ECCanvas.hpp"
#include "../renderer/ECRenderState.hpp"


ECCanvas::ECCanvas( const int& width, const int& height ) {
    pixelBuffer = ECPixelBuffer::Create( width, height );
    renderState = ECRenderState::Create();
    firstFlag = true;
}


void ECCanvas::update() {
    checkFlagAndInit();
    onUpdate();
}
void ECCanvas::draw() {
    checkFlagAndInit();
    onDraw();
    pixelBuffer->draw();
}


void ECCanvas::render( ECSmtPtr<ECModel> model, ECSmtDevPtr<ECDevLight> devLight ) {
    model->render( pixelBuffer->getDevPixels().getPtr(), renderState, pixelBuffer->getZBuffer(), devLight );
}


void ECCanvas::checkFlagAndInit() {
    if( firstFlag ) {
        firstFlag = false;
        onInit();
    }
}
