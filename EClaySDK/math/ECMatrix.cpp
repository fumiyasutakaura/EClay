#include "ECMatrix.hpp"
#include "ECVector.hpp"
#include "ECQuaternion.hpp"
#include "../etc/ECUtil.hpp"

#include <math.h>



using namespace std;

ECMatrix::ECMatrix() {
    for(int i=0; i<16; ++i) { m[i] = 0.0f; }
    m11 = 1.0f;
    m22 = 1.0f;
    m33 = 1.0f;
    m44 = 1.0f;
}
ECMatrix::ECMatrix( const float a[16] ) {
    memcpy(m, a, sizeof(float)*16);
}
ECMatrix::ECMatrix( const float a11, const float a12, const float a13, const float a14,
                   const float a21, const float a22, const float a23, const float a24,
                   const float a31, const float a32, const float a33, const float a34,
                   const float a41, const float a42, const float a43, const float a44 ) {
    m11 = a11; m12 = a12; m13= a13; m14 = a14;
    m21 = a21; m22 = a22; m23= a23; m24 = a24;
    m31 = a31; m32 = a32; m33= a33; m34 = a34;
    m41 = a41; m42 = a42; m43= a43; m44 = a44;
}
ECMatrix::ECMatrix( const ECMatrix& mat ) {
    for(int i=0; i<16; ++i) { m[i] = mat.m[i]; }
}

void ECMatrix::operator=( const ECMatrix& mat ) {
    for( int i=0; i<16; ++i ) { m[i] = mat.m[i]; }
}
ECMatrix ECMatrix::operator+( const ECMatrix& mat ) const {
    ECMatrix ret;
    for(int i=0; i<16; ++i) { ret.m[i] = m[i] + mat.m[i]; }
    return ret;
}
void ECMatrix::operator+=( const ECMatrix& mat ) {
    for(int i=0; i<16; ++i) { m[i] += mat.m[i]; }
}
ECMatrix ECMatrix::operator-( const ECMatrix& mat ) const {
    ECMatrix ret;
    for(int i=0; i<16; ++i) { ret.m[i] = m[i] - mat.m[i]; }
    return ret;
}
void ECMatrix::operator-=( const ECMatrix& mat ) {
    for(int i=0; i<16; ++i) { m[i] -= mat.m[i]; }
}
ECMatrix ECMatrix::operator*( const ECMatrix& mat ) const {
    ECMatrix ret;
    
    ret.m11 = m11*mat.m11 + m12*mat.m21 + m13*mat.m31 + m14*mat.m41;
    ret.m12 = m11*mat.m12 + m12*mat.m22 + m13*mat.m32 + m14*mat.m42;
    ret.m13 = m11*mat.m13 + m12*mat.m23 + m13*mat.m33 + m14*mat.m43;
    ret.m14 = m11*mat.m14 + m12*mat.m24 + m13*mat.m34 + m14*mat.m44;
    
    ret.m21 = m21*mat.m11 + m22*mat.m21 + m23*mat.m31 + m24*mat.m41;
    ret.m22 = m21*mat.m12 + m22*mat.m22 + m23*mat.m32 + m24*mat.m42;
    ret.m23 = m21*mat.m13 + m22*mat.m23 + m23*mat.m33 + m24*mat.m43;
    ret.m24 = m21*mat.m14 + m22*mat.m24 + m23*mat.m34 + m24*mat.m44;
    
    ret.m31 = m31*mat.m11 + m32*mat.m21 + m33*mat.m31 + m34*mat.m41;
    ret.m32 = m31*mat.m12 + m32*mat.m22 + m33*mat.m32 + m34*mat.m42;
    ret.m33 = m31*mat.m13 + m32*mat.m23 + m33*mat.m33 + m34*mat.m43;
    ret.m34 = m31*mat.m14 + m32*mat.m24 + m33*mat.m34 + m34*mat.m44;
    
    ret.m41 = m41*mat.m11 + m42*mat.m21 + m43*mat.m31 + m44*mat.m41;
    ret.m42 = m41*mat.m12 + m42*mat.m22 + m43*mat.m32 + m44*mat.m42;
    ret.m43 = m41*mat.m13 + m42*mat.m23 + m43*mat.m33 + m44*mat.m43;
    ret.m44 = m41*mat.m14 + m42*mat.m24 + m43*mat.m34 + m44*mat.m44;
    
    return ret;
}
void ECMatrix::operator*=( const ECMatrix& mat ) {
    ECMatrix temp;
    temp.m11 = m11*mat.m11 + m12*mat.m21 + m13*mat.m31 + m14*mat.m41;
    temp.m12 = m11*mat.m12 + m12*mat.m22 + m13*mat.m32 + m14*mat.m42;
    temp.m13 = m11*mat.m13 + m12*mat.m23 + m13*mat.m33 + m14*mat.m43;
    temp.m14 = m11*mat.m14 + m12*mat.m24 + m13*mat.m34 + m14*mat.m44;
    
    temp.m21 = m21*mat.m11 + m22*mat.m21 + m23*mat.m31 + m24*mat.m41;
    temp.m22 = m21*mat.m12 + m22*mat.m22 + m23*mat.m32 + m24*mat.m42;
    temp.m23 = m21*mat.m13 + m22*mat.m23 + m23*mat.m33 + m24*mat.m43;
    temp.m24 = m21*mat.m14 + m22*mat.m24 + m23*mat.m34 + m24*mat.m44;
    
    temp.m31 = m31*mat.m11 + m32*mat.m21 + m33*mat.m31 + m34*mat.m41;
    temp.m32 = m31*mat.m12 + m32*mat.m22 + m33*mat.m32 + m34*mat.m42;
    temp.m33 = m31*mat.m13 + m32*mat.m23 + m33*mat.m33 + m34*mat.m43;
    temp.m34 = m31*mat.m14 + m32*mat.m24 + m33*mat.m34 + m34*mat.m44;
    
    temp.m41 = m41*mat.m11 + m42*mat.m21 + m43*mat.m31 + m44*mat.m41;
    temp.m42 = m41*mat.m12 + m42*mat.m22 + m43*mat.m32 + m44*mat.m42;
    temp.m43 = m41*mat.m13 + m42*mat.m23 + m43*mat.m33 + m44*mat.m43;
    temp.m44 = m41*mat.m14 + m42*mat.m24 + m43*mat.m34 + m44*mat.m44;
    *this = temp;
}
ECMatrix ECMatrix::operator*( const float& a ) const {
    ECMatrix ret;
    for( int i=0; i<16; ++i ) { ret.m[i] = a * m[i]; }
    return ret;
}
void ECMatrix::operator*=( const float& a ) {
    for( int i=0; i<16; ++i ) { m[i] *= a; }
}
ECMatrix ECMatrix::operator/( const float& a ) const {
    ECMatrix ret;
    for( int i=0; i<16; ++i ) { ret.m[i] = m[i] / a; }
    return ret;
}
void ECMatrix::operator/=( const float& a ) {
    for( int i=0; i<16; ++i ) { m[i] /= a; }
}
ECVector3D ECMatrix::operator*( const ECVector3D& vec3 ) const {
    ECVector3D ret;
    float w = 1.0f;
    float converted_w = m41*vec3.x + m42*vec3.y + m43*vec3.z + m44*w;
    float reciprocal_converted_w = 1.0f;
    if( !FLOAT_EQUAL( converted_w, 0.0f ) && !FLOAT_EQUAL( converted_w, 1.0f) ) { reciprocal_converted_w = 1.0f / converted_w; }
    
    ret.x = (m11*vec3.x + m12*vec3.y + m13*vec3.z + m14*w) * reciprocal_converted_w;
    ret.y = (m21*vec3.x + m22*vec3.y + m23*vec3.z + m24*w) * reciprocal_converted_w;
    ret.z = (m31*vec3.x + m32*vec3.y + m33*vec3.z + m34*w) * reciprocal_converted_w;
    return ret;
}
ECMatrix ECMatrix::operator*( const ECQuaternion& qua ) const {
    ECMatrix ret;
    ret = (*this) * qua.getMatrix();
    return ret;
}



ECMatrix ECMatrix::Translate( const ECVector3D& vec3 ) {
    ECMatrix ret;
    ret.m14 = vec3.x;
    ret.m24 = vec3.y;
    ret.m34 = vec3.z;
    return ret;
}
ECMatrix ECMatrix::Translate( const float& tx, const float& ty, const float& tz ) {
    ECMatrix ret;
    ret.m14 = tx;
    ret.m24 = ty;
    ret.m34 = tz;
    return ret;
}

ECMatrix ECMatrix::Scale( const float s ) {
    ECMatrix ret;
    ret.m11 *= s;
    ret.m22 *= s;
    ret.m33 *= s;
    return ret;
}
ECMatrix ECMatrix::Scale( const float sx, const float sy, const float sz ) {
    ECMatrix ret;
    ret.m11 *= sx;
    ret.m22 *= sy;
    ret.m33 *= sz;
    return ret;
}
ECMatrix ECMatrix::Scale( const ECVector3D sVec ) {
    ECMatrix ret;
    ret.m11 *= sVec.x;
    ret.m22 *= sVec.y;
    ret.m33 *= sVec.z;
    return ret;
}


ECMatrix ECMatrix::RotateX( const float& degree) {
    ECMatrix ret;
    float radian = degree2radian( degree );
    ret.m22 = cos( radian );
    ret.m23 = -sin( radian );
    ret.m32 = sin( radian);
    ret.m33 = cos( radian );
    return ret;
}
ECMatrix ECMatrix::RotateY( const float& degree) {
    ECMatrix ret;
    float radian = degree2radian( degree );
    ret.m11 = cos( radian );
    ret.m13 = sin( radian );
    ret.m31 = -sin( radian );
    ret.m33 = cos( radian );
    return ret;
}
ECMatrix ECMatrix::RotateZ( const float& degree) {
    ECMatrix ret;
    float radian = degree2radian( degree );
    ret.m11 = cos( radian );
    ret.m12 = -sin( radian );
    ret.m21 = sin( radian );
    ret.m22 = cos( radian );
    return ret;
}


ECMatrix ECMatrix::Projection( const int screenW, const int screenH,
                              const float angleOfView_Degree,
                              const float near, const float far ) {
    ECMatrix ret;
    const float cot = - 1.0f / tanf( degree2radian(angleOfView_Degree*0.5) );
    const float reciprocal_far_near = 1.0f / (far - near);
    ret.m11 = ((float)screenH / (float)screenW) * cot;
    ret.m22 = cot;
    ret.m33 = far * reciprocal_far_near;
    ret.m34 = -far * near * reciprocal_far_near;
    ret.m43 = -1.0f;
    ret.m44 = 0.0f;
    return ret;
}

ECMatrix ECMatrix::Screen( const int screenW, const int screenH, const float near, const float far ) {
    ECMatrix ret;
    const float halfW = ((float)screenW) * 0.5f;
    const float halfH = ((float)screenH) * 0.5f;
    ret.m11 = -halfW;
    ret.m22 = -halfH;
    ret.m33 = - (far - near);
    ret.m14 = halfW;
    ret.m24 = halfH;
    ret.m34 = near;
    return ret;
}



void ECMatrix::print() const {
    printf(" |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n"
           ,m11,m12,m13,m14,
           m21,m22,m23,m24,
           m31,m32,m33,m34,
           m41,m42,m43,m44);
}
void ECMatrix::print( string tag ) const {
    printf("%s\n |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n |%f %f %f %f|\n"
           ,tag.c_str(),
           m11,m12,m13,m14,
           m21,m22,m23,m24,
           m31,m32,m33,m34,
           m41,m42,m43,m44);
}


float det( const ECMatrix& mat ) {

    float ret = 0.0f;
    
    ret = mat.m[0] * mat.m[5] * mat.m[10] * mat.m[15]  +  mat.m[0] * mat.m[6] * mat.m[11] * mat.m[13]  +  mat.m[0] * mat.m[7] * mat.m[9] * mat.m[14]
    + mat.m[1] * mat.m[4] * mat.m[11] * mat.m[14]  +  mat.m[1] * mat.m[6] * mat.m[8] * mat.m[15]   +  mat.m[1] * mat.m[7] * mat.m[10] * mat.m[12]
    + mat.m[2] * mat.m[4] * mat.m[9] * mat.m[15]   +  mat.m[2] * mat.m[5] * mat.m[11] * mat.m[12]  +  mat.m[2] * mat.m[7] * mat.m[8] * mat.m[13]
    + mat.m[3] * mat.m[4] * mat.m[10] * mat.m[13]  +  mat.m[3] * mat.m[5] * mat.m[8] * mat.m[14]   +  mat.m[3] * mat.m[6] * mat.m[9] * mat.m[12]
    - mat.m[0] * mat.m[5] * mat.m[11] * mat.m[14]  -  mat.m[0] * mat.m[6] * mat.m[9] * mat.m[15]   -  mat.m[0] * mat.m[7] * mat.m[10] * mat.m[13]
    - mat.m[1] * mat.m[4] * mat.m[10] * mat.m[15]  -  mat.m[1] * mat.m[6] * mat.m[11] * mat.m[12]  -  mat.m[1] * mat.m[7] * mat.m[8] * mat.m[14]
    - mat.m[2] * mat.m[4] * mat.m[11] * mat.m[13]  -  mat.m[2] * mat.m[5] * mat.m[8] * mat.m[15]   -  mat.m[2] * mat.m[7] * mat.m[9] * mat.m[12]
    - mat.m[3] * mat.m[4] * mat.m[9] * mat.m[14]   -  mat.m[3] * mat.m[5] * mat.m[10] * mat.m[12]  -  mat.m[3] * mat.m[6] * mat.m[8] * mat.m[13];
    
    return ret;
}
ECMatrix ECMatrix::inverse() const {
    
    float determinant = det( *this );
    
    if( FLOAT_EQUAL(determinant, 0.0f) ) {
        return ECMatrix();
    }
    
    ECMatrix ret;
    ret.m[0] = m[5] * m[10] * m[15]  +  m[6] * m[11] * m[13]  +  m[7] * m[9] * m[14]  -  m[5] * m[11] * m[14]  -  m[6] * m[9] * m[15]  -  m[7] * m[10] * m[13];
	ret.m[1]  = m[1] * m[11] * m[14]  +  m[2] * m[9] * m[15]  +  m[3] * m[10] * m[13]  -  m[1] * m[10] * m[15]  -  m[2] * m[11] * m[13]  -  m[3] * m[9] * m[14];
	ret.m[2]  = m[1] * m[6] * m[15]  +  m[2] * m[7] * m[13]  +  m[3] * m[5] * m[14]  -  m[1] * m[7] * m[14]  -  m[2] * m[5] * m[15]  -  m[3] * m[6] * m[13];
	ret.m[3]  = m[1] * m[7] * m[10]  +  m[2] * m[5] * m[11]  +  m[3] * m[6] * m[9]  -  m[1] * m[6] * m[11]  -  m[2] * m[7] * m[9]  -  m[3] * m[5] * m[10];
	ret.m[4]  = m[4] * m[11] * m[14]  +  m[6] * m[8] * m[15]  +  m[7] * m[10] * m[12]  -  m[4] * m[10] * m[15]  -  m[6] * m[11] * m[12]  -  m[7] * m[8] * m[14];
	ret.m[5]  = m[0] * m[10] * m[15]  +  m[2] * m[11] * m[12]  +  m[3] * m[8] * m[14]  -  m[0] * m[11] * m[14]  -  m[2] * m[8] * m[15]  -  m[3] * m[10] * m[12];
	ret.m[6]  = m[0] * m[7] * m[14]  +  m[2] * m[4] * m[15]  +  m[3] * m[6] * m[12]  -  m[0] * m[6] * m[15]  -  m[2] * m[7] * m[12]  -  m[3] * m[4] * m[14];
	ret.m[7]  = m[0] * m[6] * m[11]  +  m[2] * m[7] * m[8]  +  m[3] * m[4] * m[10]  -  m[0] * m[7] * m[10]  -  m[2] * m[4] * m[11]  -  m[3] * m[6] * m[8];
	ret.m[8]  = m[4] * m[9] * m[15]  +  m[5] * m[11] * m[12]  +  m[7] * m[8] * m[13]  -  m[4] * m[11] * m[13]  -  m[5] * m[8] * m[15]  -  m[7] * m[9] * m[12];
	ret.m[9]  = m[0] * m[11] * m[13]  +  m[1] * m[8] * m[15]  +  m[3] * m[9] * m[12]  -  m[0] * m[9] * m[15]  -  m[1] * m[11] * m[12]  -  m[3] * m[8] * m[13];
	ret.m[10] = m[0] * m[5] * m[15]  +  m[1] * m[7] * m[12]  +  m[3] * m[4] * m[13]  -  m[0] * m[7] * m[13]  -  m[1] * m[4] * m[15]  -  m[3] * m[5] * m[12];
	ret.m[11] = m[0] * m[7] * m[9]  +  m[1] * m[4] * m[11]  +  m[3] * m[5] * m[8]  -  m[0] * m[5] * m[11]  -  m[1] * m[7] * m[8]  -  m[3] * m[4] * m[9];
	ret.m[12] = m[4] * m[10] * m[13]  +  m[5] * m[8] * m[14]  +  m[6] * m[9] * m[12]  -  m[4] * m[9] * m[14]  -  m[5] * m[10] * m[12]  -  m[6] * m[8] * m[13];
	ret.m[13] = m[0] * m[9] * m[14]  +  m[1] * m[10] * m[12]  +  m[2] * m[8] * m[13]  -  m[0] * m[10] * m[13]  -  m[1] * m[8] * m[14]  -  m[2] * m[9] * m[12];
	ret.m[14] = m[0] * m[6] * m[13]  +  m[1] * m[4] * m[14]  +  m[2] * m[5] * m[12]  -  m[0] * m[5] * m[14]  -  m[1] * m[6] * m[12]  -  m[2] * m[4] * m[13];
	ret.m[15] = m[0] * m[5] * m[10]  +  m[1] * m[6] * m[8]  +  m[2] * m[4] * m[9]  -  m[0] * m[6] * m[9]  -  m[1] * m[4] * m[10]  -  m[2] * m[5] * m[8];

    ret = ret / determinant;
    
    return ret;
}

ECMatrix ECMatrix::transposition() const {
    ECMatrix ret;
    ret.m[0] = m[0]; ret.m[1] = m[4]; ret.m[2] = m[8]; ret.m[3] = m[12];
    ret.m[4] = m[1]; ret.m[5] = m[5]; ret.m[6] = m[9]; ret.m[7] = m[13];
    ret.m[8] = m[2]; ret.m[9] = m[6]; ret.m[10] = m[10]; ret.m[11] = m[14];
    ret.m[12] = m[3]; ret.m[13] = m[7]; ret.m[14] = m[11]; ret.m[15] = m[15];
    return ret;
}
