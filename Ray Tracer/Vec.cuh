#pragma once
template <typename T>
class Vec4 {
public:
	Vec4() {}
	Vec4(T x, T y, T z, T w) {
		e[0] = x; e[1] = y; e[2] = z; e[3] = w;
	}

	inline const T x() const {
		return e[0];
	}
	inline const T y() const {
		return e[1];
	}
	inline const T z() const {
		return e[2];
	}
	inline const T w() const {
		return e[3];
	}

	__host__ __device__ T operator[] (int i)const {
		return e[i];
	}


	T e[4];
};


template <typename T>
class Vec3 {
public:
	Vec3() {e[0] = 0; e[1] = 0; e[2] = 0;}
	Vec3(T x, T y, T z) {
		e[0] = x; e[1] = y; e[2] = z;
	}

	inline const T x() const {
		return e[0];
	}
	inline const T y() const {
		return e[1];
	}
	inline const T z() const {
		return e[2];
	}
	
	__host__ __device__ T Dot(const Vec3<T> &v) const { return e[0] * v[0] + e[1] * v[1] + e[2] * v[2];}
  __host__ __device__ Vec3<T> Cross(const Vec3<T> &v){
        return Vec3<T>(e[1] * v[2] - e[2] * v[1], e[2] * v[0] - e[0] * v[2], e[0] * v[1] - e[1] * v[0]);
    }
    __host__ __device__ T length()const{return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);}
    __host__ __device__ void Normalize(){
        if(this->length() > 0){
            T inverseLength = 1 / sqrt(Dot(*this));
            e[0] *= inverseLength; e[1] *= inverseLength; e[2] *= inverseLength;
        }
    }

    
    void Print(){
        std::cout << e[0] << ", " << e[1] << ", " << e[2] << std::endl;
    }

	
	
	
	 __host__ __device__ T operator[](int i) const{
		return e[i];
	}

	T e[3];
};
