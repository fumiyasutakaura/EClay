#ifndef EC_UTIL_HPP
#define EC_UTIL_HPP

#define degree2radian(x) ((x) * 0.017453292519)
#define radian2degree(x) ((x) * 57.29577951471)
#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define FLOAT_EQUAL(a,b) (ABS(a-b) < __FLT_EPSILON__)
#include <float.h>
#define FLOAT_MAX (FLT_MAX)

#define min3(a,b,c) ( (a<=b&&a<=c)?(a):((b<=c)?b:c) )
#define max3(a,b,c) ( (a>=b&&a>=c)?(a):((b>=c)?b:c) )

#include <string>
#include <vector>

float degree0To360( const float& degree );

std::string DeleteNewLineChar( const std::string& str );
std::string ReduceSpaceChar( const std::string& str );
std::vector<std::string> Separate( const std::string& str, const std::string& separater );

class ECFPSCounter  {
public:
    static ECFPSCounter* GetInstance();
	void print( const int& skip );
private:
    ECFPSCounter(){}
    ECFPSCounter(const ECFPSCounter& rhs);
    ECFPSCounter& operator=(const ECFPSCounter& rhs);
};

#endif
