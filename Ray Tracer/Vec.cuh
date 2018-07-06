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
	Vec3() {}
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
	
	 __host__ __device__ T operator[](int i) const{
		return e[i];
	}

	T e[3];
};