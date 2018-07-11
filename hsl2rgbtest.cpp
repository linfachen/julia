#include "hsl2rgb.hpp"
#include <stdio.h>

int main()
 
{
 
	COLOR_RGB  rgb={254, 216, 166};
	COLOR_HSL  hsl;
 
	
 
	RGBtoHSL(&rgb, &hsl);
	printf("H=%.3f; S=%.3f; L=%.3f\n", hsl.h, hsl.s, hsl.l);
 
 
	HSLtoRGB(&hsl, &rgb);

 
	printf("R=%d; G=%d; B=%d\n", rgb.r, rgb.g, rgb.b);
 

	return 0;
}
 