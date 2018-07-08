#pragma once
#include <vector>
#include "Vec.cuh"
template <typename T>
class VBO {
public:
	VBO(){}

	std::vector<unsigned int> indices;
	std::vector<Vec3<T>> vertices;
	Vec4<T> *Color;
};

template <typename T>
class d_VBO {
public:
	d_VBO(){}
	d_VBO(Vec3<T> *vertices, unsigned int *indices, Vec4<T> *Color) {
		this->vertices = vertices;
		this->indices = indices;
		this->Color = Color;
	}

	d_VBO(VBO<T> vbo) {
		indices = vbo.indices.data();
		vertices = vbo.vertices.data();
		Color = vbo.Color;
		N = vbo.indices.size();
	}

	unsigned int N;
	unsigned int* indices;
	Vec3<T> *vertices;
	Vec4<T> *Color;
};