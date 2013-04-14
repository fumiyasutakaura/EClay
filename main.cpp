//
//  main.cpp
//  EClay001
//
//  Created by Fumiyasu Takaura on 12/1/12.
//
//

#include <iostream>

#ifdef __APPLE__
#include <GLUT/GLUT.h>
#else
#include <GL/glut.h>
#endif

#include "EClaySDK/EClay.hpp"
#include "SampleApplication.hpp"

using namespace std;

#define SCREEN_W (480)
#define SCREEN_H (320)


extern "C" {
    void gpuInit( int argc, char** argv );
    void gpuExit( int argc, char** argv );
}


static ECSmtPtr<ECCanvas> canvas;

void draw() {
    if( !canvas.isNull() ) {
        canvas->draw();
    }
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels( SCREEN_W, SCREEN_H, GL_RGBA, GL_FLOAT, canvas->getPixelBuffer()->getPixels().getPtr() );
    glFlush();
}

void update() {
    if( !canvas.isNull() ) {
        canvas->update();
    }
    glutPostRedisplay();
}

static int _argc;
static char** _argv;
void atExitFunc() {
    canvas = (ECCanvas*)NULL;
    gpuExit( _argc, _argv );
}

int main( int argc, char** argv ) {
    
    _argc = argc;
    _argv = argv;
    atexit( atExitFunc );
    
    canvas = MainCanvas::Create( SCREEN_W, SCREEN_H );
    
    gpuInit( argc, argv );
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize( SCREEN_W, SCREEN_H );
    glutCreateWindow("EClay");
    glutDisplayFunc(draw);
    glutIdleFunc( update );
    glutMainLoop();
    
    gpuExit( argc, argv );
    
    return 0;
}
