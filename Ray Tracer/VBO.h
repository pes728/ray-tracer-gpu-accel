#pragma once
#include <vector>
#include "Vec.cuh"
template <typename T>
class VBO {
public:
	VBO(){}
	void addVec(Vec3<T> vector) {
		vertices.push_back(vector);
	}
	void addIncices(unsigned int indice) {
		indices.push_back(indice);
	}
	std::vector<unsigned int> indices;
	std::vector<Vec3<T>> vertices;
	Vec4<T> Color;
};