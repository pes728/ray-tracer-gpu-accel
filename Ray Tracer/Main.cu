#include <SFML/Graphics.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Vec.cuh"
#include "VBO.h"
#include <cmath>


typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;
typedef VBO<float> VBOf;
typedef d_VBO<float> d_VBOf;

const unsigned int INITIALWIDTH = 600;
const unsigned int INITIALHEIGHT = 600;
unsigned int WIDTH = INITIALWIDTH;
unsigned int HEIGHT = INITIALHEIGHT;
unsigned int FOV = 90;


sf::Texture screen;

__device__ void Clear(float r, float g, float b, float a, sf::Uint8* ColorBuffer, int N) {
	for (int i = 0; i < N * 4; i += 4) {
		ColorBuffer[i + 0] = r;
		ColorBuffer[i + 1] = g;
		ColorBuffer[i + 2] = b;
		ColorBuffer[i + 3] = a;
	}
}
template <typename T>
__host__ __device__ Vec3<T> NormalOfTri(Vec3<T> a, Vec3<T> b, Vec3<T> c) {
	return (a - c).Cross(a - b);
}

__device__ float area(Vec3f a, Vec3f b, Vec3f c) {
	return abs(a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2])) / 2;
}

__device__ bool Intersect(Vec3f Pos, Vec3f Vec, float t, d_VBOf vbo) {
	for (int i = 0; i < vbo.N; i += 3){	Vec3f n = NormalOfTri(vbo.vertices[vbo.indices[i]], vbo.vertices[vbo.indices[i + 1]], vbo.vertices[vbo.indices[i + 2]]);
		float d = ((vbo.vertices[vbo.indices[i]] - Pos).Dot(n)) / Vec.Dot(n);
		Vec3f point = Pos + Vec * d;

		if (area(vbo.vertices[vbo.indices[i]], vbo.vertices[vbo.indices[i + 1]], vbo.vertices[vbo.indices[i + 2]])
			== area(vbo.vertices[vbo.indices[i]], vbo.vertices[vbo.indices[i + 1]], point)
			+ area(vbo.vertices[vbo.indices[i]], point, vbo.vertices[vbo.indices[i + 2]])
			+ area(point, vbo.vertices[vbo.indices[i + 1]], vbo.vertices[vbo.indices[i + 2]]) && d > t){
			t = d;
			return true;
		}
	}
	return false;
}

__global__ void Render(sf::Uint8 *ColorBuffer, int WIDTH, int HEIGHT, float FOV ,d_VBOf *vbos, unsigned int n) {
	Clear(255,255,0,255, ColorBuffer, WIDTH * HEIGHT);
	//generate camera rays
	//camera pos = 0,0,0
	//camera vec = 1,0,0
	for(int y = 0; y < HEIGHT; y++){
		for (int x = 0; x < WIDTH; x++) {
			float t;
			for(int j = 0; j < n; j++){
				if (true || Intersect(Vec3f(), Vec3f(1, (2 * x - 1) * (WIDTH / HEIGHT) * FOV, (1 - 2 * y) * tan(FOV / 2)).Normalize(), t, vbos[j])) {
					
					ColorBuffer[x + (y * WIDTH)] = vbos[j].Color->e[0];
					ColorBuffer[x + (y * WIDTH) + 1] = vbos[j].Color->e[1];
					ColorBuffer[x + (y * WIDTH) + 2] = vbos[j].Color->e[2];
					ColorBuffer[x + (y * WIDTH) + 3] = vbos[j].Color->e[3];
				}
			}
		}
	}
}
	

int main() {
	//setup sf variables
	screen.create(WIDTH, HEIGHT);
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracer", sf::Style::Close | sf::Style::Resize);
	sf::Sprite mSprite;
	mSprite.setTexture(screen);
	sf::Event evnt;

	d_VBOf* d_vbo;
	sf::Uint8* ColorBuffer, *d_ColorBuffer;

	ColorBuffer = new sf::Uint8[WIDTH * HEIGHT * 4];

	cudaMalloc(&d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4);

	
	Vec3f tri[3] = { Vec3f(0,0,2),Vec3f(0,200,2),Vec3f(200,0,2) };
	
	VBOf vbo;
	vbo.vertices.push_back(tri[0]);
	vbo.vertices.push_back(tri[1]);
	vbo.vertices.push_back(tri[2]);
	vbo.indices.push_back(0);
	vbo.indices.push_back(1);
	vbo.indices.push_back(2);

	vbo.Color = &Vec4f(0,0,255,255);
	std::vector<VBOf> objects;
	objects.push_back(vbo);

	std::vector<d_VBOf> d_objects;
	for (VBOf vbo : objects) {
		d_objects.push_back(d_VBOf(vbo));
	}


	while (window.isOpen()) {

		while (window.pollEvent(evnt)) {
			switch (evnt.type) {
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::KeyPressed:
				switch (evnt.key.code) {
				case sf::Keyboard::Escape:
					window.close();
					break;
				}
				break;
			case sf::Event::Resized:
				if (window.getSize().x < 600) {
					window.setSize(sf::Vector2u(INITIALWIDTH, HEIGHT));
					break;
				}
				if (window.getSize().y < 600) {
					window.setSize(sf::Vector2u(WIDTH, INITIALHEIGHT));
					break;
				}

				WIDTH = window.getSize().x;
				HEIGHT = window.getSize().y;
				cudaFree(d_ColorBuffer);
				cudaMalloc(&d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4);
				delete ColorBuffer;
				ColorBuffer = new sf::Uint8[WIDTH * HEIGHT * 4];
					break;
			}
		}

		sf::Clock clock;
		//render
		//alloc mem for vbo array of size n
		
		cudaMalloc(&d_vbo, sizeof(d_VBOf) * d_objects.size());

		cudaMemcpy(d_vbo, d_objects.data(), sizeof(d_VBOf) * d_objects.size(), cudaMemcpyHostToDevice);

		Render <<<1, 1 >>>(d_ColorBuffer, WIDTH, HEIGHT, tan(FOV/2), d_vbo, objects.size());
		
		cudaMemcpy(ColorBuffer, d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
		
		//push render to screen
		screen.update(ColorBuffer);

		window.draw(mSprite);

		window.display();

		std::cout << clock.restart().asMilliseconds() << std::endl;
	}
	cudaFree(d_ColorBuffer);
	return 0;
}

