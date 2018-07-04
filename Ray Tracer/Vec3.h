#pragma once
template <typename T>
class Vec3 {
public:
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

	inline T operator[](int i)const {
		return e[i];
	}

	T e[3];
};