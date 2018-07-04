#include <SFML/Graphics.hpp>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Vec4.h"
#include "VBO.h"
#include "Vec3.h"
#include <cmath>
#include "Main.h"

typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;
typedef VBO<float> VBOf;

unsigned int WIDTH = 600;
unsigned int HEIGHT = 600;
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
	if();
}


