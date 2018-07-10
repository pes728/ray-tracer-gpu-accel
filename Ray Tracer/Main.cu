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

__device__ void Clear(float r, float g, float b, float a, sf::Uint8* ColorBuffer, unsigned int n) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (int i = index * 4; i < n; i += stride) {
		ColorBuffer[i + 0] = r;
		ColorBuffer[i + 1] = g;
		ColorBuffer[i + 2] = b;
		ColorBuffer[i + 3] = a;
	}
}

__device__ void setPixel(float r, float g, float b, float a, sf::Uint8* ColorBuffer, unsigned int i) {
		ColorBuffer[i + 0] = r;
		ColorBuffer[i + 1] = g;
		ColorBuffer[i + 2] = b;
		ColorBuffer[i + 3] = a;
}

__device__ Vec3f NormalOfTri(Vec3f a, Vec3f b, Vec3f c) {
	return (a - c).Cross(a - b);
}
template <typename T>
__device__ float area(Vec3<T> a, Vec3<T> b, Vec3<T> c) {
	return abs(a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) + c[1] * (a[2] - b[2])) / 2;
}

__global__ void Render(sf::Uint8 *ColorBuffer, int WIDTH, int HEIGHT, Vec3f *VecBuffer, unsigned int *IndiceBuffer) {
	Clear(0,255,0,255, ColorBuffer, WIDTH * HEIGHT * 4);

	Vec3f point(0,0,2);

	if(area(VecBuffer[IndiceBuffer[0]], VecBuffer[IndiceBuffer[1]], VecBuffer[IndiceBuffer[2]]) == 
		area(VecBuffer[IndiceBuffer[0]], VecBuffer[IndiceBuffer[1]], point)
		+ area(VecBuffer[IndiceBuffer[0]], point, VecBuffer[IndiceBuffer[2]])
		+ area(point, VecBuffer[IndiceBuffer[1]], VecBuffer[IndiceBuffer[2]]))
		setPixel(255,255,255,255, ColorBuffer, (blockIdx.x * blockDim.x + threadIdx.x) * 4);
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

	for (int i = 0; i < WIDTH * HEIGHT * 4; i++)
		ColorBuffer[i] = 0;

	cudaMalloc(&d_ColorBuffer, sizeof(sf::Uint8) * WIDTH * HEIGHT * 4);


	Vec3f tri[3] = { Vec3f(0,0,2),Vec3f(0,200,2),Vec3f(200,0,2) };

	VBOf vbo;
	vbo.addIndice(0);
	vbo.addIndice(1);
	vbo.addIndice(2);
	vbo.addVec(tri[0]);
	vbo.addVec(tri[1]);
	vbo.addVec(tri[2]);
	vbo.Color = Vec4f(0, 255, 0, 255);

	Vec3f *VecBuffer = vbo.vertices.data();
	Vec3f *d_VecBuffer;

	unsigned int *IndiceBuffer = vbo.indices.data();
	unsigned int *d_IndiceBuffer;


	cudaMalloc(&d_VecBuffer, sizeof(Vec3f) * vbo.vertices.size());

	cudaMemcpy(d_VecBuffer, VecBuffer, sizeof(Vec3f) * vbo.vertices.size(), cudaMemcpyHostToDevice);

	cudaMalloc(&d_IndiceBuffer, sizeof(unsigned int) * vbo.indices.size());

	cudaMemcpy(d_IndiceBuffer, IndiceBuffer,sizeof(unsigned int) * vbo.indices.size(), cudaMemcpyHostToDevice);

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
		Render <<<numBlocks, blocksize >>>(d_ColorBuffer, WIDTH, HEIGHT, d_VecBuffer, d_IndiceBuffer);

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

