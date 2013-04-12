#ifndef EC_MATH_CUH
#define EC_MATH_CUH

#include "../renderer/ECPixelBuffer.hpp"
#include "ECMatrix.cuh"



__device__ float conv0_1( const float& f ) {
    return f*0.5+0.5;
}

__device__ float4 normalize2D( const float4& vec ) {
    float4 ret;
    float lengthA = sqrt(vec.x*vec.x + vec.y*vec.y);
    float recipical_lengthA = 1.0f;
    float lengthB = sqrt(vec.z*vec.z + vec.w*vec.w);
    float recipical_lengthB = 1.0f;
    if( lengthA > 0.0f ) { recipical_lengthA = 1.0f/lengthA; }
    if( lengthB > 0.0f ) { recipical_lengthB = 1.0f/lengthB; }
    ret.x = vec.x * recipical_lengthA;
    ret.y = vec.y * recipical_lengthA;
    ret.z = vec.z * recipical_lengthB;
    ret.w = vec.w * recipical_lengthB;
    return ret;
}

__device__ float4 normalize3D( const float4& vec ) {
    float4 ret;
    ret.w = 1.0f;
    float length = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
    float recipical_length = 1.0f;
    if( length > 0.0f ) { recipical_length = 1.0f/length; }
    ret.x = vec.x * recipical_length;
    ret.y = vec.y * recipical_length;
    ret.z = vec.z * recipical_length;
    return ret;
}

__device__ float4 normalize4D( const float4& vec ) {
    float4 ret;
    float length = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w);
    float recipical_length = 1.0f;
    if( length > 0.0f ) { recipical_length = 1.0f/length; }
    ret.x = vec.x * recipical_length;
    ret.y = vec.y * recipical_length;
    ret.z = vec.z * recipical_length;
    ret.w = vec.w * recipical_length;
    return ret;
}

__device__ float vec2DLength( const float4& vec ) {
    return sqrt(vec.x*vec.x + vec.y*vec.y);
}
__device__ float vec3DLength( const float4& vec ) {
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}
__device__ float vec4DLength( const float4& vec ) {
    return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w);
}
__device__ float det2D( const float4& vec1, const float4& vec2 ) {
    return vec1.x*vec2.y - vec2.x*vec1.y;
}
__device__ float dot3D( const float4& vec1, const float4& vec2 ) {
    return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}
__device__ float4 cross3D( const float4& vec1, const float4& vec2 ) {
    float4 ret;
    ret.x = vec1.y*vec2.z - vec1.z*vec2.y;
    ret.y = vec1.x*vec2.z - vec1.z*vec2.x;
    ret.z = vec1.x*vec2.y - vec1.y*vec2.x;
    ret.w = 1.0f;
    return ret;
}


//struct ECDevMatrix {
//    float4 m1;
//    float4 m2;
//    float4 m3;
//    float4 m4;
//    __device__ ECDevMatrix() {
//        m1.x = 1.0f; m1.y = 0.0f; m1.z = 0.0f; m1.w = 0.0f;
//        m2.x = 0.0f; m2.y = 1.0f; m2.z = 0.0f; m2.w = 0.0f;
//        m3.x = 0.0f; m3.y = 0.0f; m3.z = 1.0f; m3.w = 0.0f;
//        m4.x = 0.0f; m4.y = 0.0f; m4.z = 0.0f; m4.w = 1.0f;
//    }
//    __device__ ECDevMatrix( const ECDevMatrix* argM  ) {
//        m1.x = argM->m1.x; m1.y = argM->m1.y; m1.z = argM->m1.z; m1.w = argM->m1.w;
//        m2.x = argM->m2.x; m2.y = argM->m2.y; m2.z = argM->m2.z; m2.w = argM->m2.w;
//        m3.x = argM->m3.x; m3.y = argM->m3.y; m3.z = argM->m3.z; m3.w = argM->m3.w;
//        m4.x = argM->m4.x; m4.y = argM->m4.y; m4.z = argM->m4.z; m4.w = argM->m4.w;
//    }
//    ECDevMatrix( const ECMatrix& hostM ) {
//        m1.x = hostM.m11; m1.y = hostM.m12; m1.z = hostM.m13; m1.w = hostM.m14;
//        m2.x = hostM.m21; m2.y = hostM.m22; m2.z = hostM.m23; m2.w = hostM.m24;
//        m3.x = hostM.m31; m3.y = hostM.m32; m3.z = hostM.m33; m3.w = hostM.m34;
//        m4.x = hostM.m41; m4.y = hostM.m42; m4.z = hostM.m43; m4.w = hostM.m44;
//    }
//    void identify() {
//        m1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
//        m2 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
//        m3 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
//        m4 = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
//    }
//    
//    __device__ ECDevMatrix operator*( const ECDevMatrix& mat) {
//        ECDevMatrix ret;
//        ret.m1.x = m1.x*mat.m1.x + m1.y*mat.m2.x + m1.z*mat.m3.x + m1.w*mat.m4.x;
//        ret.m1.y = m1.x*mat.m1.y + m1.y*mat.m2.y + m1.z*mat.m3.y + m1.w*mat.m4.y;
//        ret.m1.z = m1.x*mat.m1.z + m1.y*mat.m2.z + m1.z*mat.m3.z + m1.w*mat.m4.z;
//        ret.m1.w = m1.x*mat.m1.w + m1.y*mat.m2.w + m1.z*mat.m3.w + m1.w*mat.m4.w;
//        
//        ret.m2.x = m2.x*mat.m1.x + m2.y*mat.m2.x + m2.z*mat.m3.x + m2.w*mat.m4.x;
//        ret.m2.y = m2.x*mat.m1.y + m2.y*mat.m2.y + m2.z*mat.m3.y + m2.w*mat.m4.y;
//        ret.m2.z = m2.x*mat.m1.z + m2.y*mat.m2.z + m2.z*mat.m3.z + m2.w*mat.m4.z;
//        ret.m2.w = m2.x*mat.m1.w + m2.y*mat.m2.w + m2.z*mat.m3.w + m2.w*mat.m4.w;
//        
//        ret.m3.x = m3.x*mat.m1.x + m3.y*mat.m2.x + m3.z*mat.m3.x + m3.w*mat.m4.x;
//        ret.m3.y = m3.x*mat.m1.y + m3.y*mat.m2.y + m3.z*mat.m3.y + m3.w*mat.m4.y;
//        ret.m3.z = m3.x*mat.m1.z + m3.y*mat.m2.z + m3.z*mat.m3.z + m3.w*mat.m4.z;
//        ret.m3.w = m3.x*mat.m1.w + m3.y*mat.m2.w + m3.z*mat.m3.w + m3.w*mat.m4.w;
//        
//        ret.m4.x = m4.x*mat.m1.x + m4.y*mat.m2.x + m4.z*mat.m3.x + m4.w*mat.m4.x;
//        ret.m4.y = m4.x*mat.m1.y + m4.y*mat.m2.y + m4.z*mat.m3.y + m4.w*mat.m4.y;
//        ret.m4.z = m4.x*mat.m1.z + m4.y*mat.m2.z + m4.z*mat.m3.z + m4.w*mat.m4.z;
//        ret.m4.w = m4.x*mat.m1.w + m4.y*mat.m2.w + m4.z*mat.m3.w + m4.w*mat.m4.w;
//        return ret;
//    }
//    
//    void print() {
//        printf(" |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n"
//               ,m1.x,m1.y,m1.z,m1.w,
//               m2.x,m2.y,m2.z,m2.w,
//               m3.x,m3.y,m3.z,m3.w,
//               m4.x,m4.y,m4.z,m4.w);
//    }
//};
//extern int GetSizeOf_ECDevMatrix() {
//    return sizeof(ECDevMatrix);
//}

__device__ float4 operator*( const ECDevMatrix* mat, const float4& vec );
__device__ float4 operator*( const ECDevMatrix& mat, const float4& vec );
__device__ float4 operator*( const float4* lhc, const float4& rhc );
__device__ void operator*=( ECPixel& lhc, const float4& rhc );
__device__ void operator*=( ECPixel& lhc, const float& rhc );
__device__ void operator*=( float4* lhc, const float4& rhc );
__device__ float4 operator*( const float4& lhc, const float& rhc );
__device__ float4 operator*( const float& lhc, const float4& rhc );
__device__ float4 operator*( const float4& lhc, const float4& rhc );
__device__ float4 operator+( const float4& lhc, const float& rhc );
__device__ float4 operator+( const float4& lhc, const float4& rhc );
__device__ float4 operator-( const float4& lhc, const float4& rhc );
__device__ float4 operator-( const float4& lhc, const float& rhc );
struct ECDevQuaternion {
    
    union {
        struct {
            float x;
            float y;
            float z;
            float w;
        };
        float4 q;
    };
    
    __device__ ECDevQuaternion() {
        x = 0.0f;
        y = 0.0f;
        z = 1.0f;
        w = 0.0f;
    }
    __device__ ECDevQuaternion( const float4& vec_and_angle_radian ) {
        const float half_angle_radian = vec_and_angle_radian.w * 0.5f;
        const float4 norVec = normalize3D( vec_and_angle_radian );
        const float4 axis = norVec * sin(half_angle_radian);
        w = cos(half_angle_radian);
        x = axis.x;
        y = axis.y;
        z = axis.z;
    }
    __device__ ECDevQuaternion( const float4& from, const float4& to ) {
        float4 norFrom = normalize3D( from );
        float4 norTo = normalize3D( to );
        float half_angle_radian = acos( dot3D(norFrom, norTo) ) * 0.5f;
        float4 axis = normalize3D( cross3D( norFrom, norTo ) ) * sin( half_angle_radian );
        w = cos(half_angle_radian);
        x = -axis.x;
        y = axis.y;
        z = axis.z;
    }
    
    __device__ float getAngleRadian() {
        return acos(w) * 2.0f;
    }
    __device__ float getAngleDegree() {
        float half_angle_radian = acos(w);
        return half_angle_radian * 2.0f * 57.29577951471;
    }
    
    __device__ ECDevQuaternion operator*( const ECDevQuaternion& qua ) {
        ECDevQuaternion ret;
        ret.w = w * qua.w - x * qua.x - y * qua.y - z * qua.z;
        ret.x = w * qua.x + qua.w * x + y * qua.z - qua.y * z;
        ret.y = w * qua.y + qua.w * y + z * qua.x - qua.z * x;
        ret.z = w * qua.z + qua.w * z + x * qua.y - qua.x * y;
        return ret;
    }
    __device__ ECDevQuaternion operator-() {
        ECDevQuaternion ret = make_float4(x,y,z,w);
        ret.w = -ret.w;
        return ret;
    }
    
    __device__ ECDevMatrix getMatrix() const {
        ECDevMatrix ret;
        ret.m1.x = 1.0f-2*(y*y + z*z);
        ret.m1.y = 2.0f*(x*y - w*z);
        ret.m1.z = 2.0f*(x*z + w*y);
        ret.m1.w = 0.0f;
        
        ret.m2.x = 2.0f*(x*y + w*z);
        ret.m2.y = 1.0f-2.0f*(x*x + z*z);
        ret.m2.z = 2.0f*(y*z - w*x);
        ret.m2.w = 0.0f;
        
        ret.m3.x = 2.0f*(x*z - w*y);
        ret.m3.y = 2.0f*(y*z + w*x);
        ret.m3.z = 1.0f-2.0f*(x*x + y*y);
        ret.m3.w = 0.0f;
        
        ret.m4.x = 0.0f;
        ret.m4.y = 0.0f;
        ret.m4.z = 0.0f;
        ret.m4.w = 1.0f;
        return ret;
    }
    
    __device__ ECDevMatrix operator*( const ECDevMatrix& mat ) {
        ECDevMatrix ret;
        ret = getMatrix() * mat;
        return ret;
    }
    
    __device__ float4 operator*( const float4& vec ) {
        float4 ret;
        ret = getMatrix() * vec;
        return ret;
    }
    
};

__device__ ECDevMatrix operator*( const ECDevMatrix& mat, const ECDevQuaternion& qua ) {
    ECDevMatrix ret;
    ECDevMatrix a = mat;
    ECDevMatrix b = qua.getMatrix();
    ret = a * b;
    return ret;
}

__device__ ECDevMatrix getRotateMatrix( const ECDevMatrix* argM, const float4* scale ) {
    ECDevMatrix ret(argM);
    ret.m1.w = 0.0f;
    ret.m2.w = 0.0f;
    ret.m3.w = 0.0f;
    ret.m1.x /= scale->x;
    ret.m2.y /= scale->y;
    ret.m3.z /= scale->z;
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////
__device__ float4 multMatVecNoDivW( const ECDevMatrix* mat, const float4& vec ) {
    float4 ret;
    ret.x = mat->m1.x*vec.x + mat->m1.y*vec.y + mat->m1.z*vec.z + mat->m1.w*vec.w;
    ret.y = mat->m2.x*vec.x + mat->m2.y*vec.y + mat->m2.z*vec.z + mat->m2.w*vec.w;
    ret.z = mat->m3.x*vec.x + mat->m3.y*vec.y + mat->m3.z*vec.z + mat->m3.w*vec.w;
    ret.w = mat->m4.x*vec.x + mat->m4.y*vec.y + mat->m4.z*vec.z + mat->m4.w*vec.w;
    return ret;
}
__device__ float4 operator/( const float4& f4, const float& f ) {
    float4 ret;
    ret.x = f4.x / f;
    ret.y = f4.y / f;
    ret.z = f4.z / f;
    ret.w = f4.w / f;
    return ret;
}
///////////////////////////////////////////////////////////////////////////////////////////
__device__ float4 operator*( const ECDevMatrix* mat, const float4& vec ) {
    float4 ret;
    ret.w = mat->m4.x*vec.x + mat->m4.y*vec.y + mat->m4.z*vec.z + mat->m4.w*vec.w;
    float reciprocal_w = 1.0f/ret.w;
    ret.x = (mat->m1.x*vec.x + mat->m1.y*vec.y + mat->m1.z*vec.z + mat->m1.w*vec.w)*reciprocal_w;
    ret.y = (mat->m2.x*vec.x + mat->m2.y*vec.y + mat->m2.z*vec.z + mat->m2.w*vec.w)*reciprocal_w;
    ret.z = (mat->m3.x*vec.x + mat->m3.y*vec.y + mat->m3.z*vec.z + mat->m3.w*vec.w)*reciprocal_w;
    ret.w = 1.0f;
    return ret;
}
__device__ float4 operator*( const ECDevMatrix& mat, const float4& vec ) {
    float4 ret;
    ret.w = mat.m4.x*vec.x + mat.m4.y*vec.y + mat.m4.z*vec.z + mat.m4.w*vec.w;
    float reciprocal_w = 1.0f/ret.w;
    ret.x = (mat.m1.x*vec.x + mat.m1.y*vec.y + mat.m1.z*vec.z + mat.m1.w*vec.w)*reciprocal_w;
    ret.y = (mat.m2.x*vec.x + mat.m2.y*vec.y + mat.m2.z*vec.z + mat.m2.w*vec.w)*reciprocal_w;
    ret.z = (mat.m3.x*vec.x + mat.m3.y*vec.y + mat.m3.z*vec.z + mat.m3.w*vec.w)*reciprocal_w;
    ret.w = 1.0f;
    return ret;
}
__device__ float4 operator*( const float4* lhc, const float4& rhc ) {
    float4 ret;
    ret.x = lhc->x * rhc.x;
    ret.y = lhc->y * rhc.y;
    ret.z = lhc->z * rhc.z;
    ret.w = lhc->w * rhc.w;
    return ret;
}
__device__ void operator*=( ECPixel& lhc, const float4& rhc ) {
    lhc.r *= rhc.x;
    lhc.g *= rhc.y;
    lhc.b *= rhc.z;
    lhc.a *= rhc.w;
}
__device__ void operator*=( ECPixel& lhc, const float& rhc ) {
    lhc.r *= rhc;
    lhc.g *= rhc;
    lhc.b *= rhc;
    lhc.a *= rhc;
}
__device__ void operator*=( float4* lhc, const float4& rhc ) {
    lhc->x *= rhc.x;
    lhc->y *= rhc.y;
    lhc->z *= rhc.z;
    lhc->w *= rhc.w;
}
__device__ float4 operator*( const float4& lhc, const float& rhc ) {
    float4 ret;
    ret.x = lhc.x * rhc;
    ret.y = lhc.y * rhc;
    ret.z = lhc.z * rhc;
    ret.w = lhc.w * rhc;
    return ret;
}
__device__ float4 operator*( const float& lhc, const float4& rhc ) {
    float4 ret;
    ret.x = lhc * rhc.x;
    ret.y = lhc * rhc.y;
    ret.z = lhc * rhc.z;
    ret.w = lhc * rhc.w;
    return ret;
}
__device__ float4 operator*( const float4& lhc, const float4& rhc ) {
    float4 ret;
    ret.x = lhc.x * rhc.x;
    ret.y = lhc.y * rhc.y;
    ret.z = lhc.z * rhc.z;
    ret.w = lhc.w * rhc.w;
    return ret;
}
__device__ float4 operator+( const float4& lhc, const float& rhc ) {
    float4 ret;
    ret.x = lhc.x + rhc;
    ret.y = lhc.y + rhc;
    ret.z = lhc.z + rhc;
    ret.w = lhc.w + rhc;
    return ret;
}
__device__ float4 operator+( const float4& lhc, const float4& rhc ) {
    float4 ret;
    ret.x = lhc.x + rhc.x;
    ret.y = lhc.y + rhc.y;
    ret.z = lhc.z + rhc.z;
    ret.w = lhc.w + rhc.w;
    return ret;
}
__device__ float4 operator-( const float4& lhc, const float4& rhc ) {
    float4 ret;
    ret.x = lhc.x - rhc.x;
    ret.y = lhc.y - rhc.y;
    ret.z = lhc.z - rhc.z;
    ret.w = lhc.w - rhc.w;
    return ret;
}
__device__ float4 operator-( const float4& lhc, const float& rhc ) {
    float4 ret;
    ret.x = lhc.x - rhc;
    ret.y = lhc.y - rhc;
    ret.z = lhc.z - rhc;
    ret.w = lhc.w - rhc;
    return ret;
}
__device__ float4 operator-( const float4& f ) {
    float4 ret;
    ret.x = -f.x;
    ret.y = -f.y;
    ret.z = -f.z;
    ret.w = -f.w;
    return ret;
}



/*
__device__ bool devLeftPosition( const float4* point, const float4& start, const float4& end ) {
    const float perpendicularLineLength =  point->x * (start.y - end.y) + start.x * (end.y - point->y) + end.x * (point->y - start.y);
    if( perpendicularLineLength >= 0.0f ) { return true; }
    return false;
}
__device__ bool devInTheTriangleCounterClockwise( const float4 *point,
                                                 const float4 v1,
                                                 const float4 v2,
                                                 const float4 v3) {
    if(devLeftPosition( point, v1, v2 ) &&
       devLeftPosition( point, v2, v3 ) &&
       devLeftPosition( point, v3, v1 )   ){
        return true;
    }
    return false;
}
*/
__device__ bool devInTheTriangleCounterClockwise(const float4 *point,
                                                 const float4 &v1,
                                                 const float4 &v2,
                                                 const float4 &v3) {
    return (
            point->x * (v1.y - v2.y) + v1.x * (v2.y - point->y) + v2.x * (point->y - v1.y) >= 0.0f &&
            point->x * (v2.y - v3.y) + v2.x * (v3.y - point->y) + v3.x * (point->y - v2.y) >= 0.0f &&
            point->x * (v3.y - v1.y) + v3.x * (v1.y - point->y) + v1.x * (point->y - v3.y) >= 0.0f
            )?true:false;
       
}





#endif
