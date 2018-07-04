#pragma once
#include <vector>
#include "Vec3.h"
#include "Vec4.h"
template <typename T>
class VBO {
public:
	VBO(){}
	void add(Vec3<T> vector) {
		vertexBufferObj.push_back(vector);
	}
	std::vector<Vec3<T>> vertexBufferObj;
	Vec4<T> Color;
};