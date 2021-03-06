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

unsigned int blocksize = 256;
unsigned int numBlocks = ceil((WIDTH * HEIGHT + blocksize - 1) / blocksize);

sf::Texture screen;

__global__ void Clear(float r, float g, float b, float a, sf::Uint8* ColorBuffer, unsigned int n) {
		unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int stride = blockDim.x * gridDim.x;
		for(int i = index * 4; i < n; i +=stride){
			ColorBuffer[i + 0] = r;
			ColorBuffer[i + 1] = g;
			ColorBuffer[i + 2] = b;
			ColorBuffer[i + 3] = a;
		}
}

__device__ Vec3f NormalOfTri(Vec3f a, Vec3f b, Vec3f c) {
	return (a - c).Cross(a - b);
}
template <typename T>
__device__ float4 area(const Vec3<T> &a, const Vec3<T> &b, const Vec3<T> &c) {
	return abs(a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2])) / 2;
}

__device__ void Intersect(Vec3f Pos, Vec3f Vec, float &t, d_VBOf* d_vbo) {
area(d_vbo->vertices[0], d_vbo->vertices[1], d_vbo->vertices[2]);
	return;
}

__global__ void Render(sf::Uint8 *ColorBuffer, int WIDTH, int HEIGHT, d_VBOf *d_vbo) {
	float t = INFINITY;
}


sf::Uint8* ColorBuffer;

int main() {
	//setup sf variables
	screen.create(WIDTH, HEIGHT);
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Ray Tracer", sf::Style::Close | sf::Style::Resize);
	sf::Sprite mSprite;
	mSprite.setTexture(screen);
	sf::Event evnt;

	sf::Uint8* ColorBuffer, *d_ColorBuffer;

	ColorBuffer = new sf::Uint8[WIDTH * HEIGHT * 4];

	for(int i = 0; i < WIDTH * HEIGHT * 4; i++)
	ColorBuffer[i] = 0;

	cudaMalloc(&d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4);

	
	Vec3f tri[3] = { Vec3f(0,0,2),Vec3f(0,200,2),Vec3f(200,0,2) };

	VBOf vbo;
	vbo.addIndices(0);
	vbo.addIndices(1);
	vbo.addIndices(2);
	vbo.addVec(tri[0]);
	vbo.addVec(tri[1]);
	vbo.addVec(tri[2]);
	vbo.Color = Vec4f(0,255,0,255);

	d_VBOf *d_vbo;

	d_VBOf host_vbo (vbo);

	cudaMalloc(&d_vbo, sizeof(d_VBOf));

	cudaMemcpy(d_vbo, &host_vbo, sizeof(host_vbo), cudaMemcpyHostToDevice);

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
				cudaFree(ColorBuffer);
				cudaMalloc(&ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4);
				unsigned int numBlocks = ceil((WIDTH * HEIGHT + blocksize - 1) / blocksize);
					break;
			}
		}

		sf::Clock clock;
		//render
		Clear<<<numBlocks, blocksize>>>(0,255,0,255,d_ColorBuffer, WIDTH * HEIGHT * 4);
		Render <<<1, 1 >>>(d_ColorBuffer, WIDTH, HEIGHT, d_vbo);
		
		cudaMemcpy(ColorBuffer, d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
		
		//push render to screen
		screen.update(ColorBuffer);

		window.clear(sf::Color::Black);

		window.draw(mSprite);

		window.display();

		std::cout << clock.restart().asMilliseconds() << std::endl;
	}
	cudaFree(d_ColorBuffer);
	return 0;
}

