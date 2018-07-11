
#define min3v(v1, v2, v3)	((v1)>(v2)? ((v2)>(v3)?(v3):(v2)):((v1)>(v3)?(v3):(v2)))
#define max3v(v1, v2, v3)	((v1)<(v2)? ((v2)<(v3)?(v3):(v2)):((v1)<(v3)?(v3):(v1)))
 
  
typedef struct
{
	unsigned char r;				// [0,255]
	unsigned char g;			    // [0,255]
	unsigned char b;				// [0,255]
}COLOR_RGB;
 
 
typedef struct
{
	float h;				// [0,360]
	float s;		        // [0,100]
	float l;		        // [0,100]
}COLOR_HSL;
 
 
// Converts RGB to HSL
static void RGBtoHSL(const COLOR_RGB *rgb, COLOR_HSL *hsl)
{
 
	float h=0, s=0, l=0;
 
	// normalizes red-green-blue values
	float r = rgb->r/255.0;
	float g = rgb->g/255.0;
	float b = rgb->b/255.0;
 
 
 
	float maxVal = max3v(r, g, b);
	float minVal = min3v(r, g, b);
 
 
 
	// hue
	if(maxVal == minVal)
	{
		h = 0; // undefined
	}
 
	else if(maxVal==r && g>=b)
	{
		h = 60.0*(g-b)/(maxVal-minVal);
	}
 
	else if(maxVal==r && g<b)
	{
		h = 60.0*(g-b)/(maxVal-minVal) + 360.0;
	}
 
	else if(maxVal==g)
	{
		h = 60.0*(b-r)/(maxVal-minVal) + 120.0;
	}
 
	else if(maxVal==b)
	{
		h = 60.0*(r-g)/(maxVal-minVal) + 240.0;
	}
 
 
	// luminance
	l = (maxVal+minVal)/2.0f;
 
 
	// saturation
	if(l == 0 || maxVal == minVal)
	{
		s = 0;
	}
 
	else if(0<l && l<=0.5)
	{
		s = (maxVal-minVal)/(maxVal+minVal);
	}
 
	else if(l>0.5)
	{
		s = (maxVal-minVal)/(2 - (maxVal+minVal)); //(maxVal-minVal > 0)?
	}
 
 
	hsl->h = (h>360)? 360 : ((h<0)?0:h); 
	hsl->s = ((s>1)? 1 : ((s<0)?0:s))*100;
	hsl->l = ((l>1)? 1 : ((l<0)?0:l))*100;
} 
 
 
// Converts HSL to RGB
static void HSLtoRGB(const COLOR_HSL *hsl, COLOR_RGB *rgb) 
{
 
	float h = hsl->h;					// h must be [0, 360]
	float s = hsl->s/100.0;	            // s must be [0, 1]
	float l = hsl->l/100.0;		        // l must be [0, 1]
 
	float R, G, B;
 
	if(hsl->s == 0)
	{
		// achromatic color (gray scale)
		R = G = B = l*255.0;
	}
	else
	{
		float q = (l<0.5)?(l * (1.0+s)):(l+s - (l*s));
		float p = (2.0 * l) - q;
 
		float Hk = h/360.0;
		float T[3];
 
		T[0] = Hk + 0.3333333;	// Tr	0.3333333f=1.0/3.0
		T[1] = Hk;				// Tb
		T[2] = Hk - 0.3333333;	// Tg
 
 
		for(int i=0; i<3; i++)
		{
			if(T[i] < 0) T[i] += 1.0;
			if(T[i] > 1) T[i] -= 1.0;
 
 
			if((T[i]*6) < 1)
			{
				T[i] = p + ((q-p)*6.0*T[i]);
			}
			else if((T[i]*2.0) < 1) //(1.0/6.0)<=T[i] && T[i]<0.5
			{
				T[i] = q;
			}
 
			else if((T[i]*3.0) < 2) // 0.5<=T[i] && T[i]<(2.0/3.0)
			{
				T[i] = p + (q-p) * ((2.0/3.0) - T[i]) * 6.0;
			}
			else T[i] = p;
		}
 
 
		R = T[0]*255.0;
		G = T[1]*255.0;
		B = T[2]*255.0;
	}
 
	rgb->r = (unsigned char)((R>255)? 255 : ((R<0)?0 : R));
	rgb->g = (unsigned char)((G>255)? 255 : ((G<0)?0 : G));
	rgb->b = (unsigned char)((B>255)? 255 : ((B<0)?0 : B));
 
}
 
 
 
 
 

 
 
