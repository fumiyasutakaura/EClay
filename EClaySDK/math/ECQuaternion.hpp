#ifndef EC_QUATERNION_HPP
#define EC_QUATERNION_HPP

#include <string>

class ECMatrix;
class ECVector3D;



class ECQuaternion {
public:
    union {
        float q[4];
        struct {
            float x;
            float y;
            float z;
            float w;
        };
    };
    
    ECQuaternion();
    ECQuaternion( const float& ax, const float& ay, const  float& az, const float& rw );
    ECQuaternion( const ECVector3D& vec, const float& angle_degree );
    ECQuaternion( const ECVector3D& from, const ECVector3D& to );
    
    float getAngleRadian() const;
    float getAngleDegree() const;
    
    ECMatrix getMatrix() const;
    
    ECQuaternion operator*( const ECQuaternion& qua ) const;
    ECQuaternion operator-() const;
    ECMatrix operator*( const ECMatrix& mat ) const;
    ECVector3D operator*( const ECVector3D& vec ) const;
    
    void print() const;
    void print( std::string tag ) const;
    
private:
    
};


#endif
