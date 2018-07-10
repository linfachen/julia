#include <ostream>
#include <iostream>

#ifdef  __KERNEL__

#else
#define __device__
#endif


class MyComplex{
public:
    float _x;
    float _y;    
        
	__device__ MyComplex():_x(0),_y(0){}
	__device__ MyComplex(float x,float y):_x(x),_y(y){}
    
    //operator+
	__device__ MyComplex operator+(const MyComplex& other){
        return  MyComplex(_x+other._x,_y+other._y);
    } 
    //operator-
	__device__ MyComplex operator-(const MyComplex& other){
        return  MyComplex(_x-other._x,_y-other._y);
    }
    //operator*    
	__device__ MyComplex operator*(const MyComplex& other){
        return  MyComplex(_x*other._x-_y*other._y,_x*other._y+_y*other._x);
    } 

	__device__ MyComplex conjugate(){
        return MyComplex(_x,-_y);
    }
    
    //_x*_x+_y*_y
	__device__ float mod(){
        return _x*_x+_y*_y;
    }
   


	friend std::ostream& operator<<(std::ostream & os, MyComplex &obj) {
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

	friend std::ostream& operator<<(std::ostream & os, MyComplex &&obj) {
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

