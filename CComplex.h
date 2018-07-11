#include <ostream>
#include <iostream>

#ifdef  __KERNEL__
#define __DEV__ __device__
#else
#define __DEV__
#endif


class CComplex{
public:
    float _x;
    float _y;    
        
	__DEV__ CComplex():_x(0),_y(0){}
	__DEV__ CComplex(float x,float y):_x(x),_y(y){}
    
    //operator+
	__DEV__ CComplex operator+(const CComplex& other){
        return  CComplex(_x+other._x,_y+other._y);
    } 
    //operator-
	__device__ CComplex operator-(const CComplex& other){
        return  CComplex(_x-other._x,_y-other._y);
    }
    //operator*    
	__DEV__ CComplex operator*(const CComplex& other){
        return  CComplex(_x*other._x-_y*other._y,_x*other._y+_y*other._x);
    } 

	__DEV__ CComplex conjugate(){
        return CComplex(_x,-_y);
    }
    
    //_x*_x+_y*_y
	__DEV__ float mod(){
        return _x*_x+_y*_y;
    }
   


	friend std::ostream& operator<<(std::ostream & os, CComplex &obj) {
		if(obj._y<0){
			os << obj._x << obj._y << "i";
		}
		else if (obj._y > 0) {
			os << obj._x << "+" << obj._y << "i";
		}
		else {
			os << obj._x;
		}
		return os;
	}

	friend std::ostream& operator<<(std::ostream & os, CComplex &&obj) {
		if (obj._y<0) {
			os << obj._x << obj._y << "i";
		}
		else if (obj._y > 0) {
			os << obj._x << "+" << obj._y << "i";
		}
		else {
			os << obj._x;
		}
		return os;
	}

};

