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

const unsigned int INITIALWIDTH = 600;
const unsigned int INITIALHEIGHT = 600;
unsigned int WIDTH = INITIALWIDTH;
unsigned int HEIGHT = INITIALHEIGHT;

sf::Texture screen;

__device__ void Clear(float r, float g, float b, float a, sf::Uint8* ColorBuffer, int N) {
	for (int i = 0; i < N * 4; i += 4) {
		ColorBuffer[i + 0] = r;
		ColorBuffer[i + 1] = g;
		ColorBuffer[i + 2] = b;
		ColorBuffer[i + 3] = a;
	}
}

__global__ void Render(sf::Uint8 *ColorBuffer, int WIDTH, int HEIGHT, std::vector<VBOf> VBOs) {
	Clear(255,0,0,255, ColorBuffer, N);
	for(int y = 0; y <HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float t;
			for(VBOf vbo:VBOs){
				//cameraPos = 0,0,0
				//cameraVec = 0,0,1
				if(Intersect(Vec3f(), Vec3f(0,0,1) , &t,vbo)){
					ColorBuffer[x + (y * WIDTH)] = vbo.color[0];
					ColorBuffer[x + (y * WIDTH) + 1] = vbo.color[1];
					ColorBuffer[x + (y * WIDTH) + 2] = vbo.color[2];
					ColorBuffer[x + (y * WIDTH) + 3] = vbo.color[3];
				}
			}
		}
	}
	
	__device__ float area(Vec3f a, Vec3f b, Vec3f c){
		return abs(a[1] * (b[2] - c[2]) + b[1] * (c[2] - a[2]) +  c[1] * (a[2] - b[2]))/2;
	}
	
	__device__ bool Intersect(Vec3f Pos, Vec3f Vec, float t, VBOf vbo)){
		if(area())
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
	vbo.addVec(tri[0]);
	vbo.addVec(tri[1]);
	vbo.addVec(tri[2]);
	vbo.addIndices(0);
	vbo.addIndices(1);
	vbo.addIndices(2);
	vbo.color = Vec4f(0,0,255,255);
	
	std::vector<VBOf> objects;
	objects.push_back(vbo);

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

					break;
			}
		}

		sf::Clock clock;
		//render

		Render <<<1, 1 >> >(d_ColorBuffer, WIDTH, HEIGHT, objects);
		
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

