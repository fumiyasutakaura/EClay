#ifndef EC_MATRIX_DEV_CUH
#define EC_MATRIX_DEV_CUH

#include <cutil.h>
#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include "ECMatrix.hpp"
#include "../smart_pointer/ECSmtPtr.hpp"


struct ECDevMatrix {
    float4 m1;
    float4 m2;
    float4 m3;
    float4 m4;
    __device__ ECDevMatrix() {
        m1.x = 1.0f; m1.y = 0.0f; m1.z = 0.0f; m1.w = 0.0f;
        m2.x = 0.0f; m2.y = 1.0f; m2.z = 0.0f; m2.w = 0.0f;
        m3.x = 0.0f; m3.y = 0.0f; m3.z = 1.0f; m3.w = 0.0f;
        m4.x = 0.0f; m4.y = 0.0f; m4.z = 0.0f; m4.w = 1.0f;
    }
    __device__ ECDevMatrix( const ECDevMatrix* argM  ) {
        m1.x = argM->m1.x; m1.y = argM->m1.y; m1.z = argM->m1.z; m1.w = argM->m1.w;
        m2.x = argM->m2.x; m2.y = argM->m2.y; m2.z = argM->m2.z; m2.w = argM->m2.w;
        m3.x = argM->m3.x; m3.y = argM->m3.y; m3.z = argM->m3.z; m3.w = argM->m3.w;
        m4.x = argM->m4.x; m4.y = argM->m4.y; m4.z = argM->m4.z; m4.w = argM->m4.w;
    }
    ECDevMatrix( const ECMatrix& hostM ) {
        m1.x = hostM.m11; m1.y = hostM.m12; m1.z = hostM.m13; m1.w = hostM.m14;
        m2.x = hostM.m21; m2.y = hostM.m22; m2.z = hostM.m23; m2.w = hostM.m24;
        m3.x = hostM.m31; m3.y = hostM.m32; m3.z = hostM.m33; m3.w = hostM.m34;
        m4.x = hostM.m41; m4.y = hostM.m42; m4.z = hostM.m43; m4.w = hostM.m44;
    }
    void identify() {
        m1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
        m2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
        m3 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        m4 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    __device__ ECDevMatrix operator*( const ECDevMatrix& mat) {
        ECDevMatrix ret;
        ret.m1.x = m1.x*mat.m1.x + m1.y*mat.m2.x + m1.z*mat.m3.x + m1.w*mat.m4.x;
        ret.m1.y = m1.x*mat.m1.y + m1.y*mat.m2.y + m1.z*mat.m3.y + m1.w*mat.m4.y;
        ret.m1.z = m1.x*mat.m1.z + m1.y*mat.m2.z + m1.z*mat.m3.z + m1.w*mat.m4.z;
        ret.m1.w = m1.x*mat.m1.w + m1.y*mat.m2.w + m1.z*mat.m3.w + m1.w*mat.m4.w;
        
        ret.m2.x = m2.x*mat.m1.x + m2.y*mat.m2.x + m2.z*mat.m3.x + m2.w*mat.m4.x;
        ret.m2.y = m2.x*mat.m1.y + m2.y*mat.m2.y + m2.z*mat.m3.y + m2.w*mat.m4.y;
        ret.m2.z = m2.x*mat.m1.z + m2.y*mat.m2.z + m2.z*mat.m3.z + m2.w*mat.m4.z;
        ret.m2.w = m2.x*mat.m1.w + m2.y*mat.m2.w + m2.z*mat.m3.w + m2.w*mat.m4.w;
        
        ret.m3.x = m3.x*mat.m1.x + m3.y*mat.m2.x + m3.z*mat.m3.x + m3.w*mat.m4.x;
        ret.m3.y = m3.x*mat.m1.y + m3.y*mat.m2.y + m3.z*mat.m3.y + m3.w*mat.m4.y;
        ret.m3.z = m3.x*mat.m1.z + m3.y*mat.m2.z + m3.z*mat.m3.z + m3.w*mat.m4.z;
        ret.m3.w = m3.x*mat.m1.w + m3.y*mat.m2.w + m3.z*mat.m3.w + m3.w*mat.m4.w;
        
        ret.m4.x = m4.x*mat.m1.x + m4.y*mat.m2.x + m4.z*mat.m3.x + m4.w*mat.m4.x;
        ret.m4.y = m4.x*mat.m1.y + m4.y*mat.m2.y + m4.z*mat.m3.y + m4.w*mat.m4.y;
        ret.m4.z = m4.x*mat.m1.z + m4.y*mat.m2.z + m4.z*mat.m3.z + m4.w*mat.m4.z;
        ret.m4.w = m4.x*mat.m1.w + m4.y*mat.m2.w + m4.z*mat.m3.w + m4.w*mat.m4.w;
        return ret;
    }
    
    void set( const ECMatrix& hM ) {
        m1.x = hM.m11; m1.y = hM.m12; m1.z = hM.m13; m1.w = hM.m14;
        m2.x = hM.m21; m2.y = hM.m22; m2.z = hM.m23; m2.w = hM.m24;
        m3.x = hM.m31; m3.y = hM.m32; m3.z = hM.m33; m3.w = hM.m34;
        m4.x = hM.m41; m4.y = hM.m42; m4.z = hM.m43; m4.w = hM.m44;
    }
    
    void print() {
        printf(" |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n"
               ,m1.x,m1.y,m1.z,m1.w,
               m2.x,m2.y,m2.z,m2.w,
               m3.x,m3.y,m3.z,m3.w,
               m4.x,m4.y,m4.z,m4.w);
    }
};



#endif

