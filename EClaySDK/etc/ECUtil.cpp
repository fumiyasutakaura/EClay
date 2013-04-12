#include "ECUtil.hpp" 

#include <time.h>
#include <iostream>

using namespace std;

float degree0To360( const float& degree ) {
    float ret = 0.0f;
    float temp = degree;
    if( temp > 0.0f ) {
        while ( temp >= 360.0f ) {
            temp -= 360.0f;
        }
    }
    else {
        while ( temp <= -360.0f ) {
            temp += 360.0f;
        }
        temp += 360.0f;
    }
    ret = temp;
    return ret;
}

string DeleteNewLineChar( const string& str ) {
    string ret = "";
    for( int i=0; i<str.size(); ++i ) {
        const char c = str[i];
        if( c != '\n' ) { ret += c; }
    }
    return ret;
}

string ReduceSpaceChar( const string& str ) {
    string ret = "";
    int count = 0;
    while ( count < str.size() ) {
        string lastChar = "";
        int lastIndex = (int)ret.size() -1;
        if( lastIndex >= 0 ) { lastChar = ret[lastIndex]; }
        if( lastChar == " " ) {
            if( str[count] != ' ' ) { ret += str[count]; }
        }
        else {
            if( str[count] == ' ' ) {
                if( ret.size() > 0 && count != str.size()-1 ) { ret += str[count]; }
            }
            else { ret += str[count]; }
        }
        ++count;
    }
    if( ret.size() > 0 ) {
        if( ret[ret.size()-1] == ' ' ) {
            ret = ret.substr( 0, ret.size()-1 );
        }
    }
    return ret;
}
vector<string> Separate( const string& str, const string& separater ) {
//    vector<string> ret;
//    string temp = "";
//    int count = 0;
//    while ( count < str.size() ) {
//        if( str[count] != separater[0] ) {
//            temp += str[count];
//        }
//        else {
//            ret.push_back( temp );
//            temp = "";
//        }
//        ++count;
//    }
//    if( temp != "" ) {
//        ret.push_back( temp );
//    }
//    return ret;
    
//    vector<string> ret;
//    string::size_type index = str.find( separater );
//    if( index != string::npos ) {
//        ret.push_back( str.substr(0,index) );
//        vector<string> back = Separate( str.substr(index+separater.size()), separater );
//        if( back.size() ) {
//            vector<string>::iterator it = back.begin();
//            while( it != back.end() ) {
//                ret.push_back( *it );
//                ++it;
//            }
//        }
//    }
//    else {
//        if( str.size() ) {
//            ret.push_back( str );
//        }
//    }
//    return ret;
    
    vector<string> ret;
    string::size_type index = str.find( separater );
    if( index == string::npos ) {
        if( str.size() ) {
            ret.push_back( str );
        }
    }
    else {
        string tempStr = str;
        while ( index != string::npos ) {
            ret.push_back( tempStr.substr(0,index) );
            tempStr = tempStr.substr(index+separater.size());
            index = tempStr.find( separater );
        }
        if( tempStr.substr(0,index).size() ) {
            ret.push_back( tempStr.substr(0,index) );
        }
    }
    return ret;

}


ECFPSCounter* ECFPSCounter::GetInstance() {
    static ECFPSCounter instance;
    return &instance;
}
void ECFPSCounter::print( const int& skip )
{
	static clock_t last_time = clock();
	
	static int print_cnt = 0;
	if( ++print_cnt % skip == 0 ){
		printf("FPS: %f\n", (float)CLOCKS_PER_SEC / (clock() - last_time) );
	}
    
	last_time = clock();
}
