// same but let's start with fixed screen and compute res, vga resolution of 640x480
// same but now adding openmp
#include <iostream>
#include <complex>
#include <omp.h>
#include <SDL.h>
#undef main

using cplx = std::complex<double>;
using namespace std::literals;

constexpr int height{ 480 };
constexpr int width{ 640 };
int pitch = width * 4;
double cap{ 2 };
int max_iters{ 50 };
uint8_t scale_val = ((uint8_t)-1) / max_iters;
// data for plane plotting, messy for now:
double x_extent{ 3.0 };
double z_extent{ 2.0 };
double x_cent{ -0.6 };
double z_cent{ 0.0 };

cplx fc(cplx c, cplx z = 0. + 0.i)
{
	return z * z + c;
}

int iter_div(cplx c)
{
	// given starting point c, returns the number of iterations it took before diverging
	// past upper bound, and max_iters if it didn't diverge
	// (this way we can quickly increase max_iters, only need iterate over values that previously failed to diverge, 
	// the ones that diverged earlier will still do so)
	cplx eval = fc(c);
	for (int i = 0; i < max_iters; i++)
	{
		if (std::abs(eval) > cap)
		{
			return i;
		}
		eval = fc(c, eval);
	}
	return max_iters;
}

void init_complex_plane(cplx* plane)
{
	double step_x{ x_extent / static_cast<double>(width) };
	double step_z{ z_extent / static_cast<double>(height) };
	for (int z = 0; z < height; z++)
	{
#pragma omp parallel for default(none) shared(plane) schedule(static, 8)
		for (int x = 0; x < width; x++)
		{
			cplx val{ x_cent - x_extent / 2.0 + step_x * x, z_cent - z_extent / 2.0 + step_z * z };
			plane[z*width + x] = val;
		}
	}
}

void mandelbrot_diverge_pixels(uint32_t* pixels_div, cplx* mat)
{
	// computes iter_div and formats as pixel color in one passthru
// can't enable the below command until msvc supports openMP 3.0, but likely causes cache problems since large block jumps
//#pragma omp parallel for collapse(2) default(none) shared(mat, pixels_div) schedule(static, 8)
	for (int z = 0; z < height; z++)
	{
#pragma omp parallel for default(none) shared(mat, pixels_div) schedule(static, 8)
		for (int x = 0; x < width; x++)
		{
			int pos = z * width + x;
			uint8_t single_col = scale_val * (max_iters - iter_div(mat[pos]));
			// repeat shifted to the r g and b positions
			uint32_t color = (single_col << 24) + (single_col << 16) + (single_col << 8) + single_col;
			pixels_div[pos] = color;
		}
	}
}

void compute_and_render(SDL_Texture* texture, uint32_t* pixels_div, cplx* mat, bool to_shift = false)
{
	if (to_shift)
	{
		init_complex_plane(mat);
	}
	SDL_LockTexture(texture, NULL, (void**)&pixels_div, &pitch);
	mandelbrot_diverge_pixels(pixels_div, mat);
	SDL_UnlockTexture(texture);
}

void resize_compute_region(int x1, int z1, int x2, int z2)
{
	// first rescale x,z,1,2 relative current origin and extent
	// first sort to get min and max, NOTE the x axis is ok but z inverted so order flipped here
	if (x2 < x1)
	{
		int temp = x1;
		x1 = x2;
		x2 = temp;
	}
	if (z2 < z1)
	{
		int temp = z1;
		z1 = z2;
		z2 = temp;
	}
	// now rescale into the actuall coord system, them finally compute new extent and cent from that
	double new_x1 = ((x1 / static_cast<double>(width)) - 0.5) * x_extent + x_cent;
	double new_x2 = ((x2 / static_cast<double>(width)) - 0.5) * x_extent + x_cent;
	double new_z1 = ((z1 / static_cast<double>(height)) - 0.5) * z_extent + z_cent;
	double new_z2 = ((z2 / static_cast<double>(height)) - 0.5) * z_extent + z_cent;
	x_extent = new_x2 - new_x1;
	z_extent = new_z2 - new_z1;
	x_cent = (new_x2 + new_x1) / 2.0;
	z_cent = (new_z2 + new_z1) / 2.0;
}

int main()
{
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window* win;
	SDL_Renderer* rend;
	SDL_Texture* texture;

	win = SDL_CreateWindow("Mandelbrot",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		width, height,
		SDL_WINDOW_SHOWN);
	rend = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
	texture = SDL_CreateTexture(rend,
		SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STREAMING,
		width,
		height);

	// now update the texture with color vals
	// NOTE here we use unified array for pixels and diverge, makes computing faster
	cplx* plane = new cplx[width * height];
	uint32_t* pixels_div = new uint32_t[width * height];

	compute_and_render(texture, pixels_div, plane, true);

	SDL_RenderClear(rend);
	SDL_RenderCopy(rend, texture, NULL, NULL);
	SDL_RenderPresent(rend);

	bool quit = false;
	int xdown = 0;
	int ydown = 0;
	int xup = 0;
	int yup = 0;
	SDL_Event e;
	while (!quit)
	{
		if (SDL_WaitEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
			else if (e.type == SDL_KEYDOWN)
			{
				switch (e.key.keysym.sym)
				{
				case SDLK_UP:
					// here we want to increase the cap by 10%
					cap = cap * 1.2;
					std::cout << "cap: " << cap << std::endl;
					compute_and_render(texture, pixels_div, plane);
					std::cout << "updated" << std::endl;
					break;
				case SDLK_DOWN:
					// here we want to increase the cap by 10%
					cap = cap * 0.8;
					std::cout << "cap: " << cap << std::endl;
					compute_and_render(texture, pixels_div, plane);
					std::cout << "updated" << std::endl;
					break;
				case SDLK_LEFT:
					max_iters = static_cast<int>(max_iters * 0.8);
					scale_val = ((uint8_t)-1) / max_iters;
					std::cout << "max iters: " << max_iters << std::endl;
					compute_and_render(texture, pixels_div, plane);
					std::cout << "updated" << std::endl;
					break;
				case SDLK_RIGHT:
					max_iters = static_cast<int>(max_iters * 1.2);
					scale_val = ((uint8_t)-1) / max_iters;
					std::cout << "max iters: " << max_iters << std::endl;
					compute_and_render(texture, pixels_div, plane);
					std::cout << "updated" << std::endl;
					break;
				case SDLK_SPACE:
					// reset plot params here
					x_extent = 3.0;
					z_extent = 2.0;
					x_cent = -0.6;
					z_cent = 0.0;
					cap = 2;
					max_iters = 50;
					scale_val = ((uint8_t)-1) / max_iters;
					compute_and_render(texture, pixels_div, plane, true);
					std::cout << "plot params reset" << std::endl;
					break;
				default:
					continue;
				}
			}
			else if (e.type == SDL_MOUSEBUTTONDOWN)
			{
				// gives pixel coord from top left to bottom right
				// if we don't scale outer window should be pretty easy to convert to complex coords
				SDL_GetMouseState(&xdown, &ydown);
			}
			else if (e.type == SDL_MOUSEBUTTONUP)
			{
				std::cout << x_cent << " " << z_cent << " " << x_extent << " " << z_extent << std::endl;
				SDL_GetMouseState(&xup, &yup);
				resize_compute_region(xdown, ydown, xup, yup);
				compute_and_render(texture, pixels_div, plane, true);
				std::cout << x_cent << " " << z_cent << " " << x_extent << " " << z_extent << std::endl;
				std::cout << "updated" << std::endl;
			}
		}
		SDL_RenderClear(rend);
		SDL_RenderCopy(rend, texture, NULL, NULL);
		SDL_RenderPresent(rend);
	}

	return 0;
}