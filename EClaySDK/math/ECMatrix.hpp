#ifndef EC_MATRIX_HPP
#define EC_MATRIX_HPP

#include <string>


class ECQuaternion;
class ECVector3D;


class ECMatrix {
public:
    union {
        float m[16];
        struct {
            float m11; float m12; float m13; float m14;
            float m21; float m22; float m23; float m24;
            float m31; float m32; float m33; float m34;
            float m41; float m42; float m43; float m44;
        };
    };

    ECMatrix();
    ECMatrix( const float a[16] );
    ECMatrix( const float a11, const float a12, const float a13, const float a14,
             const float a21, const float a22, const float a23, const float a24,
             const float a31, const float a32, const float a33, const float a34,
             const float a41, const float a42, const float a43, const float a44 );
    ECMatrix( const ECMatrix& mat );
    
    
    void operator=( const ECMatrix& mat );
    ECMatrix operator+( const ECMatrix& mat ) const;
    void operator+=( const ECMatrix& mat );
    ECMatrix operator-( const ECMatrix& mat ) const;
    void operator-=( const ECMatrix& mat );
    ECMatrix operator*( const ECMatrix& mat ) const;
    void operator*=( const ECMatrix& mat );
    ECMatrix operator*( const float& a ) const;
    void operator*=( const float& a );
    ECMatrix operator/( const float& a ) const;
    void operator/=( const float& a );
    ECVector3D operator*( const ECVector3D& vec3 ) const;
    ECMatrix operator*( const ECQuaternion& qua ) const;
    
    static ECMatrix Translate( const ECVector3D& vec3 );
    static ECMatrix Translate( const float& tx, const float& ty, const float& tz );
    
    static ECMatrix Scale( const float s );
    static ECMatrix Scale( const float sx, const float sy, const float sz );
    static ECMatrix Scale( const ECVector3D sVec );
    
    static ECMatrix RotateX( const float& degree);
    static ECMatrix RotateY( const float& degree);
    static ECMatrix RotateZ( const float& degree);
    
    static ECMatrix Projection( const int screenW, const int screenH,
                               const float angleOfView_Degree,
                               const float near, const float far );
    
    static ECMatrix Screen( const int screenW, const int screenH, const float near, const float far );

    
    
    void print() const;
    void print( std::string tag ) const;
    
    ECMatrix inverse() const;
    ECMatrix transposition() const;
private:
    
};

static ECMatrix operator*( const float& a, const ECMatrix& mat) {
    ECMatrix ret;
    for( int i=0; i<16; ++i ) { ret.m[i] = mat.m[i] * a; }
    return ret;
}

static float det( const ECMatrix& mat );

#endif

