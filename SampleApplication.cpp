#include "SampleApplication.hpp"


void MainCanvas::onInit() {
    
    // dev Ligth
    ECDevLight *hostLight = (ECDevLight*)malloc( sizeof(ECDevLight) );
    {
        hostLight->position = make_float4(0.0f, 0.0f, -500.0f, 1.0f);
        hostLight->direction = make_float4( 0.0f, 0.0f, -1.0f, 1.0f );
        hostLight->ambient = make_float4( 0.1f, 0.1f, 0.1f, 1.0f );
        hostLight->diffuse = make_float4( 0.8f, 0.8f, 0.8f, 1.0f );
        hostLight->specular = make_float4( 1.0f, 1.0f, 1.0f, 1.0f );
        devLight = ECSmtDevPtr<ECDevLight>(hostLight,sizeof(ECDevLight));
    }
    free( hostLight );
    
    // make model
    model = ECWaveFrontObject::Create( "../res/majo.obj" );
    /*
    model = ECWaveFrontObject::Create( "../res/hare.obj" );
    model = ECWaveFrontObject::Create( "../res/ninteru03.obj" );
    model = ECWaveFrontObject::Create( "../res/manjuu.obj" );
    model = ECWaveFrontObject::Create( "../res/bomb.obj" );
    model = ECWaveFrontObject::Create( "../res/onigiri.obj" );
     */
    ECQuaternion q1(ECVector3D(0,1,0),180);
    ECMatrix modelWorldMat = (ECMatrix::Translate(0,0.0f,-200) * q1);
    model->setWorldMatrix( modelWorldMat );
    const float scaleVal = 30.0f;
    model->setScale( make_float4(scaleVal,scaleVal,scaleVal,1.0f) );
    
    // fps counter skip flame
    fpscounter_skip_flame = 60;
    
    // render state
    const int angleOfView_Degree = 60;
    const int near = -100;
    const int far = -600;
    renderState->setAngleOfView_Degree( angleOfView_Degree );
    renderState->setNear( near );
    renderState->setFar( far );
    renderState->setScreenWidth( pixelBuffer->getWidth() );
    renderState->setScreenHeight( pixelBuffer->getHeight() );
    
}

void MainCanvas::onUpdate() {
    
    static float time = 0.0;
    
    if( !model.isNull() ) {
        model->setWorldMatrix( model->getWorldMatrix() * ECQuaternion(ECVector3D(0,1,0),(time-0.1)*5) *ECQuaternion(ECVector3D(0,1,0),-time*5) );
    }
    
    time += 0.1;
    
}

void MainCanvas::onDraw() {
    
    pixelBuffer->clear();
    
    if( !model.isNull() ) {
        render( model, devLight );
    }
    
    ECFPSCounter::GetInstance()->print( fpscounter_skip_flame );
    
}



