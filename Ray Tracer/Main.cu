#include <SFML/Graphics.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Vec4.h"
#include "VBO.h"
#include "Vec3.h"
#include <cmath>


typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;
typedef VBO<float> VBOf;

const unsigned int INITIALWIDTH = 600;
const unsigned int INITIALHEIGHT = 600;
unsigned int WIDTH = INITIALWIDTH;
unsigned int HEIGHT = INITIALHEIGHT;

sf::Texture screen;
sf::Uint8* ColorBuffer;

void Clear(Vec4f ClearColor) {
	for (register int i = 0; i < WIDTH*HEIGHT * 4; i += 4) {
			ColorBuffer[i + 0] = ClearColor.x();
			ColorBuffer[i + 1] = ClearColor.y();
			ColorBuffer[i + 2] = ClearColor.z();
			ColorBuffer[i + 3] = ClearColor.w();
	}
}

__global__ void Render(sf::Uint8 *ColorBuffer) {
	
}

int main() {
	//setup sf variables
	screen.create(WIDTH, HEIGHT);
	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Rasterizer", sf::Style::Close | sf::Style::Resize);
	sf::Sprite mSprite;
	mSprite.setTexture(screen);
	sf::Event evnt;

	cudaMallocManaged(&ColorBuffer, WIDTH * HEIGHT * 4);

	Vec3f tri[3] = { Vec3f(0,0,2),Vec3f(0,200,2),Vec3f(200,0,2)};

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
				if (window.getSize().x < 600){
				window.setSize(sf::Vector2u(INITIALWIDTH, HEIGHT));
				 break;
				 }
				if (window.getSize().y < 600){
				window.setSize(sf::Vector2u(WIDTH, INITIALHEIGHT));
				break;
				}


				WIDTH = window.getSize().x;
				HEIGHT = window.getSize().y;
				delete ColorBuffer;
				ColorBuffer = nullptr;
				ColorBuffer = new sf::Uint8[WIDTH*HEIGHT * 4];
				break;
			}
		}
		
		Clear(Vec4f(255,0,0,255));
		sf::Clock clock;
		//render
		Render<<<1,1>>>(ColorBuffer);


		//push render to screen
		screen.update(ColorBuffer);

		
		window.clear(sf::Color::Black);

		window.draw(mSprite);

		window.display();

		std::cout << clock.restart().asMilliseconds() << std::endl;
	}
}
