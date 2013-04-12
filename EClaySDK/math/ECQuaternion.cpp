#include "ECQuaternion.hpp"

#include "ECVector.hpp"
#include "ECMatrix.hpp"
#include "../etc/ECUtil.hpp"

#include <math.h>
#include <stdio.h>

using namespace std;




ECQuaternion::ECQuaternion() {
    x = 0.0f;
    y = 0.0f;
    z = 1.0f;
    w = 0.0f;
}
ECQuaternion::ECQuaternion( const float& ax, const float& ay, const  float& az, const float& rw ) {
    x = ax;
    y = ay;
    z = az;
    w = rw;
}
ECQuaternion::ECQuaternion( const ECVector3D& vec, const float& angle_degree ) {
    float half_angle_radian = degree2radian( angle_degree ) * 0.5f;
    ECVector3D axis = vec.getNormal() * sin(half_angle_radian);
    w = cos(half_angle_radian);
    x = axis.x;
    y = axis.y;
    z = axis.z;
}
ECQuaternion::ECQuaternion( const ECVector3D& from, const ECVector3D& to ) {
    ECVector3D norFrom = from.getNormal();
    ECVector3D norTo = to.getNormal();
    float half_angle_radian = acos( norFrom.dot(norTo) ) * 0.5f;
    ECVector3D axis = ( norTo.cross( norFrom ) ).getNormal() * sin( half_angle_radian );
    w = cos(half_angle_radian);
    x = axis.x;
    y = axis.y;
    z = axis.z;
}


float ECQuaternion::getAngleRadian() const {
    return acos(w) * 2.0f;
}
float ECQuaternion::getAngleDegree() const {
    float half_angle_radian = acos(w);
    return radian2degree( half_angle_radian * 2.0f );
}

ECMatrix ECQuaternion::getMatrix() const {
    ECMatrix ret;
    ret.m[0] = 1-2*(y*y + z*z); ret.m[1] = 2*(x*y - w*z);   ret.m[2] = 2*(x*z + w*y);
    ret.m[4] = 2*(x*y + w*z);   ret.m[5] = 1-2*(x*x + z*z); ret.m[6] = 2*(y*z - w*x);
    ret.m[8] = 2*(x*z - w*y);   ret.m[9] = 2*(y*z + w*x);   ret.m[10] = 1-2*(x*x + y*y);
    return ret;
}


ECQuaternion ECQuaternion::operator*( const ECQuaternion& qua ) const {
    ECQuaternion ret;
    ret.w = w * qua.w - x * qua.x - y * qua.y - z * qua.z;
    ret.x = w * qua.x + qua.w * x + y * qua.z - qua.y * z;
    ret.y = w * qua.y + qua.w * y + z * qua.x - qua.z * x;
    ret.z = w * qua.z + qua.w * z + x * qua.y - qua.x * y;
    return ret;
}
ECQuaternion ECQuaternion::operator-() const {
    ECQuaternion ret = *this;
    ret.w = -ret.w;
    return ret;
}
ECMatrix ECQuaternion::operator*( const ECMatrix& mat ) const {
    ECMatrix ret;
    ret = getMatrix() * mat;
    return ret;
}
ECVector3D ECQuaternion::operator*( const ECVector3D& vec ) const {
    ECVector3D ret;
    ret = getMatrix() * vec;
    return ret;
}

void ECQuaternion::print() const {
    printf("Q( (%f, %f, %f) %f )\n",x,y,z,w);
}
void ECQuaternion::print( string tag ) const {
    printf("%s Q( (%f, %f, %f) %f )\n",tag.c_str(),x,y,z,w);
}

