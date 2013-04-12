#ifndef EC_VECTOR_HPP
#define EC_VECTOR_HPP

#include <string>



class ECVector2D {
public:
    union {
        struct {
            float x;
            float y;
        };
        struct {
            float u;
            float v;
        };
    };
    
    ECVector2D();
    ECVector2D( const float a[2] );
    ECVector2D( const float ax, const float ay );
    
    void operator=( const ECVector2D& vec );
    void operator+=( const ECVector2D& vec );
    ECVector2D operator+( const ECVector2D& vec ) const;
    void operator-=( const ECVector2D& vec );
    ECVector2D operator-( const ECVector2D& vec ) const;
    void operator*=( const float& a );
    ECVector2D operator*( const float& a ) const;
    void operator/=( const float& a );
    ECVector2D operator/( const float& a ) const;
    
    float dot( const ECVector2D& v ) const;
    float getLength() const;
    void normalize();
    ECVector2D getNormal() const;
    
    void print() const;
    void print( std::string tag ) const;
    
};
static ECVector2D operator*( const float& a, const ECVector2D& vec );
static float Dot( const ECVector2D& v1, const ECVector2D& v2 );

enum PointLocation {
    kPointLocationLeftSide,
    kPointLocationRightSide,
    kPointLocationOnTheLine,
};
enum Clockwise {
    kClockwise,
    kCounterClockwise,
};
static PointLocation pointLocation( const ECVector2D& point, const ECVector2D& start, const ECVector2D& end );
static bool inTheTriangle( const ECVector2D& point,
                           Clockwise clockwise,
                           const ECVector2D& v1, const ECVector2D& v2, const ECVector2D& v3 );


class ECVector3D {
public:
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float xy[2];
    };
    
    ECVector3D();
    ECVector3D( const float ax, const float ay, const float az );
    
    void operator=( const ECVector3D& vec );
    void operator+=( const ECVector3D& vec );
    ECVector3D operator+( const ECVector3D& vec ) const;
    void operator-=( const ECVector3D& vec );
    ECVector3D operator-( const ECVector3D& vec ) const;
    void operator*=( const float& a );
    ECVector3D operator*( const float& a ) const;
    void operator/=( const float& a );
    ECVector3D operator/( const float& a ) const;
    
    
    float dot( const ECVector3D& v ) const;
    ECVector3D cross( const ECVector3D& v ) const;
    float getLength() const;
    void normalize();
    ECVector3D getNormal() const;
    
    void print() const;
    void print( std::string tag ) const;
    
};
static ECVector3D operator*( const float& a, const ECVector3D& vec );
static float Dot( const ECVector3D& v1, const ECVector3D& v2 );
static ECVector3D Cross( const ECVector3D& v1, const ECVector3D& v2 );




class ECVector4D {
public:
    union {
        struct {
            float r;
            float g;
            float b;
            float a;
        };
        struct {
            float x;
            float y;
            float z;
            float w;
        };
    };
    
    ECVector4D();
    ECVector4D( const float red, const float green, const float blue, const float alpha );

    void operator+=( const ECVector4D& col );
    ECVector4D operator+( const ECVector4D& col );
    void operator-=( const ECVector4D& col );
    ECVector4D operator-( const ECVector4D& col );
    void operator*=( const float& v );
    ECVector4D operator*( const float& v );
    void operator/=( const float& v );
    ECVector4D operator/( const float& v );
    
    void print() const;
    void print( std::string tag ) const;
    
};
static ECVector4D operator*( const float& a, const ECVector4D& col );



#endif
