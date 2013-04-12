#include "ECVector.hpp"

#include "../etc/ECUtil.hpp"

#include <stdio.h>
#include <math.h>

using namespace std;



ECVector2D::ECVector2D() {
    x = 0.0f;
    y = 0.0f;
}
ECVector2D::ECVector2D( const float a[2] ) {
    x = a[0];
    y = a[1];
}
ECVector2D::ECVector2D( const float ax, const float ay ) {
    x = ax;
    y = ay;
}

void ECVector2D::operator=( const ECVector2D& vec ) {
    x = vec.x;
    y = vec.y;
}
void ECVector2D::operator+=( const ECVector2D& vec ) {
    x += vec.x;
    y += vec.y;
}
ECVector2D ECVector2D::operator+( const ECVector2D& vec ) const {
    ECVector2D ret;
    ret.x = x + vec.x;
    ret.y = y + vec.y;
    return ret;
}
void ECVector2D::operator-=( const ECVector2D& vec ) {
    x -= vec.x;
    y -= vec.y;
}
ECVector2D ECVector2D::operator-( const ECVector2D& vec ) const {
    ECVector2D ret;
    ret.x = x - vec.x;
    ret.y = y - vec.y;
    return ret;
}
void ECVector2D::operator*=( const float& a ) {
    x *= a;
    y *= a;
}
ECVector2D ECVector2D::operator*( const float& a ) const {
    ECVector2D ret;
    ret.x = x * a;
    ret.y = y * a;
    return ret;
}
void ECVector2D::operator/=( const float& a ) {
    x /= a;
    y /= a;
}
ECVector2D ECVector2D::operator/( const float& a ) const {
    ECVector2D ret;
    ret.x = x / a;
    ret.y = y / a;
    return ret;
}

float ECVector2D::dot( const ECVector2D& v ) const {
    return x*v.x + y*v.y;
}

float ECVector2D::getLength() const {
    return sqrtf(x*x + y*y);
}

void ECVector2D::normalize() {
    float denominator = sqrtf(x*x + y*y);
    float reciprocal_length = 1.0f;
    if( denominator > 0.0f ) { reciprocal_length = 1.0f / denominator; }
    x *= reciprocal_length;
    y *= reciprocal_length;
}
ECVector2D ECVector2D::getNormal() const {
    ECVector2D ret;
    float denominator = sqrtf(x*x + y*y);
    float reciprocal_length = 1.0f;
    if( denominator > 0.0f ) { reciprocal_length = 1.0f / denominator; }
    ret.x = x * reciprocal_length;
    ret.y = y * reciprocal_length;
    return ret;
}

void ECVector2D::print() const {
    printf("(%f, %f)\n",x,y);
}
void ECVector2D::print( string tag ) const {
    printf("%s (%f, %f)\n",tag.c_str(),x,y);
}

ECVector2D operator*( const float& a, const ECVector2D& vec ) {
    ECVector2D ret;
    ret.x = a * vec.x;
    ret.y = a * vec.y;
    return ret;
}
float Dot( const ECVector2D& v1, const ECVector2D& v2 ) {
    return v1.x*v2.x + v1.y*v2.y;
}


PointLocation pointLocation( const ECVector2D& point, const ECVector2D& start, const ECVector2D& end ) {
    const float perpendicularLineLength =  point.x * (start.y - end.y) + start.x * (end.y - point.y) + end.x * (point.y - start.y);
    
    if( FLOAT_EQUAL(perpendicularLineLength, 0.0f) ) { return kPointLocationOnTheLine; }
    else if( perpendicularLineLength > 0.0f ) { return kPointLocationLeftSide; }
    else if( perpendicularLineLength < 0.0f ) { return kPointLocationRightSide; }
    return kPointLocationLeftSide;
}
bool inTheTriangle( const ECVector2D& point,
                          Clockwise clockwise,
                          const ECVector2D& v1, const ECVector2D& v2, const ECVector2D& v3 ) {
    if( clockwise == kClockwise ) {
        if( pointLocation( point, v1, v2 ) == kPointLocationLeftSide ) { return false; }
        if( pointLocation( point, v2, v3 ) == kPointLocationLeftSide ) { return false; }
        if( pointLocation( point, v3, v1 ) == kPointLocationLeftSide ) { return false; }
    }
    else {
        if( pointLocation( point, v1, v2 ) == kPointLocationRightSide ) { return false; }
        if( pointLocation( point, v2, v3 ) == kPointLocationRightSide ) { return false; }
        if( pointLocation( point, v3, v1 ) == kPointLocationRightSide ) { return false; }
    }
    
    return true;
}







ECVector3D::ECVector3D() {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}
ECVector3D::ECVector3D( const float ax, const float ay, const float az ) {
    x = ax;
    y = ay;
    z = az;
}

void ECVector3D::operator=( const ECVector3D& vec ) {
    x = vec.x;
    y = vec.y;
    z = vec.z;
}
void ECVector3D::operator+=( const ECVector3D& vec ) {
    x += vec.x;
    y += vec.y;
    z += vec.z;
}
ECVector3D ECVector3D::operator+( const ECVector3D& vec ) const {
    ECVector3D ret;
    ret.x = x + vec.x;
    ret.y = y + vec.y;
    ret.z = z + vec.z;
    return ret;
}
void ECVector3D::operator-=( const ECVector3D& vec ) {
    x -= vec.x;
    y -= vec.y;
    z -= vec.z;
}
ECVector3D ECVector3D::operator-( const ECVector3D& vec ) const {
    ECVector3D ret;
    ret.x = x - vec.x;
    ret.y = y - vec.y;
    ret.z = z - vec.z;
    return ret;
}
void ECVector3D::operator*=( const float& a ) {
    x *= a;
    y *= a;
    z *= a;
}
ECVector3D ECVector3D::operator*( const float& a ) const {
    ECVector3D ret;
    ret.x = x * a;
    ret.y = y * a;
    ret.z = z * a;
    return ret;
}
void ECVector3D::operator/=( const float& a ) {
    x /= a;
    y /= a;
    z = a;
}
ECVector3D ECVector3D::operator/( const float& a ) const {
    ECVector3D ret;
    ret.x = x / a;
    ret.y = y / a;
    ret.z = z / a;
    return ret;
}

float ECVector3D::dot( const ECVector3D& v ) const {
    return x*v.x + y*v.y + z*v.z;
}
ECVector3D ECVector3D::cross( const ECVector3D& v ) const {
    return ECVector3D( y*v.z - z*v.y, x*v.z - z*v.x, x*v.y - y*v.x);
}

float ECVector3D::getLength() const {
    return sqrtf(x*x + y*y + z*z);
}

void ECVector3D::normalize() {
    float denominator = sqrtf(x*x + y*y + z*z);
    float reciprocal_length = 1.0f;
    if( denominator > 0.0f ) { reciprocal_length = 1.0f / denominator; }
    x *= reciprocal_length;
    y *= reciprocal_length;
    z *= reciprocal_length;
}
ECVector3D ECVector3D::getNormal() const {
    ECVector3D ret;
    float denominator = sqrtf(x*x + y*y + z*z);
    float reciprocal_length = 1.0f;
    if( denominator > 0.0f ) { reciprocal_length = 1.0f / denominator; }
    ret.x = x * reciprocal_length;
    ret.y = y * reciprocal_length;
    ret.z = z * reciprocal_length;
    return ret;
}

void ECVector3D::print() const {
    printf("(%f, %f, %f)\n",x,y,z);
}

void ECVector3D::print( string tag ) const {
    printf("%s (%f, %f, %f)\n",tag.c_str(),x,y,z);
}

ECVector3D operator*( const float& a, const ECVector3D& vec ) {
    ECVector3D ret;
    ret.x = a * vec.x;
    ret.y = a * vec.y;
    ret.z = a * vec.z;
    return ret;
}
float Dot( const ECVector3D& v1, const ECVector3D& v2 ) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
ECVector3D Cross( const ECVector3D& v1, const ECVector3D& v2 ) {
    ECVector3D ret;
    ret.x = v1.y*v2.z - v1.z*v2.y;
    ret.y = v1.x*v2.z - v1.z*v2.x;
    ret.z = v1.x*v2.y - v1.y*v2.x;
    return ret;
}






ECVector4D::ECVector4D() {
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
    a = 1.0f;
}
ECVector4D::ECVector4D( const float red, const float green, const float blue, const float alpha ) {
    r = red;
    g = green;
    b = blue;
    a = alpha;
}
void ECVector4D::operator+=( const ECVector4D& col ) {
    r += col.r;
    g += col.g;
    b += col.b;
    a += col.a;
}
ECVector4D ECVector4D::operator+( const ECVector4D& col ) {
    ECVector4D ret;
    ret.r = r + col.r;
    ret.g = g + col.g;
    ret.b = b + col.b;
    ret.a = a + col.a;
    return ret;
}
void ECVector4D::operator-=( const ECVector4D& col ) {
    r -= col.r;
    g -= col.g;
    b -= col.b;
    a -= col.a;
}
ECVector4D ECVector4D::operator-( const ECVector4D& col ) {
    ECVector4D ret;
    ret.r = r - col.r;
    ret.g = g - col.g;
    ret.b = b - col.b;
    ret.a = a - col.a;
    return ret;
}
void ECVector4D::operator*=( const float& v ) {
    r *= v;
    g *= v;
    b *= v;
    a *= v;
}
ECVector4D ECVector4D::operator*( const float& v ) {
    ECVector4D ret;
    ret.r = r * v;
    ret.g = g * v;
    ret.b = b * v;
    ret.a = a * v;
    return ret;
}
void ECVector4D::operator/=( const float& v ) {
    r /= v;
    g /= v;
    b /= v;
    a /= v;
}
ECVector4D ECVector4D::operator/( const float& v ) {
    ECVector4D ret;
    ret.r = r / v;
    ret.g = g / v;
    ret.b = b / v;
    ret.a = a / v;
    return ret;
}

void ECVector4D::print() const {
    printf("(%f, %f, %f, %f)\n",r,g,b,a);
}
void ECVector4D::print( string tag ) const {
    printf("%s (%f, %f, %f, %f)\n",tag.c_str(),r,g,b,a);
}

ECVector4D operator*( const float& a, const ECVector4D& col ) {
    ECVector4D ret;
    ret.r = a * col.r;
    ret.g = a * col.g;
    ret.b = a * col.a;
    ret.a = a * col.a;
    return ret;
}
