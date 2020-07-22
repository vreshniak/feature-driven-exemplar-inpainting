#include "stdlib.h"	/* rand, malloc */
#include "string.h" /* memcpy */
#include "stdio.h"	/* printf */
#include "time.h"	/* time */
#include "math.h"	/* fabs */
#include "omp.h"
#include "c_patchmatch.h"


#define print(a, args...) printf("%s(%s:%d) " a,  __func__,__FILE__, __LINE__, ##args)
#define println(a, args...) print(a "\n", ##args)
// Numerical recipes, Section 7.1
#define RAND(x) (4294957665 * ((x) & 4294967295) + ((x) >> 32))

typedef float float_type;

/*
	Input params
	============
	source : source image
	source_ind : linear indices in the source image
	target : target image
	target_ind : linear indices in the target image
	neighbors : indices of nearest neighbors
		on input contain initial guess
	distances : distances to nearest neighbors
	source_size : length of the source_ind array
	target_size : length of the target_ind array
*/

/*
	PRE
	===
	1) 2d arrays are in C order
*/


int im_x, im_y, im_ch, patch_size;
int *patch_h;
float_type *source, *target, *pweight, *lambda;


/* L2 weighted patch distance */
// inline float_type L2d2(int ps, int pt)
// {
// 	// #pragma omp parallel
// 	// {
// 	// fprintf(stderr,"%d\n", omp_get_thread_num());
// 	float_type dist = 0.0;
// 	for (int ch = 0; ch < im_ch; ch++){
// 		int offset_s = ch*im_x*im_y + ps;
// 		int offset_t = ch*im_x*im_y + pt;
// 		float_type *weight = pweight;
// 		int *h = patch_h;
// 		for (int i = 0; i < patch_size; i++, weight++, h++ ){
// 			int s_id = offset_s + (*h);
// 			int t_id = offset_t + (*h);
// 			dist += lambda[t_id] * (source[s_id] - target[t_id]) * (source[s_id] - target[t_id]) * (*weight);
// 		}
// 	}

// 	return dist;
// 	// }
// }
inline double L2d2(int ps, int pt)
{
	float_type *weight = pweight;
	float_type dist = 0.0;
	for (int i = 0; i < patch_size; i++, weight++ ){
		for (int ch = 0; ch < im_ch; ch++){
			int s_id = (ps + patch_h[i]) * im_ch + ch;
			int t_id = (pt + patch_h[i]) * im_ch + ch;
			dist += lambda[t_id] * (source[s_id] - target[t_id]) * (source[s_id] - target[t_id]) * (*weight);
		}
	}

	return dist;
}



void pm( float_type* source_im, char* source_mask, int* source_ind, int source_y, int source_x, int source_ch,
		 float_type* target_im, char* target_mask, int* target_ind, int target_y, int target_x, int target_ch, int target_ind_size,
		 int *neighbors, float_type *distances, int* patch_ind, int patch_ind_size, float_type* weight, float_type* lambdas, int max_rand_shots, int max_iterations, int max_window_size, float_type TOL )
{
	/* Global vars */
	im_y    = target_y;
	im_x    = target_x;
	im_ch   = target_ch;
	source  = source_im;
	target  = target_im;
	pweight = weight;
	lambda  = lambdas;
	patch_h = patch_ind;
	patch_size = patch_ind_size;

	int target_mask_size = target_x * target_y;
	int source_mask_size = source_x * source_y;

	/* Search window size */
	int window_size = (max_window_size != -1) ? max_window_size : ( (source_x>source_y) ? source_x-1 : source_y-1 );

	/* Distances for the given initial guess */
	for ( int i = 0; i < target_ind_size; i++ )
		distances[target_ind[i]] = L2d2(neighbors[target_ind[i]], target_ind[i]);


	int no_improve_iters = 0;
	float_type max_dist = 0.0, max_dist_old;
	/* Improve NNF */
	for (int iter = 0; iter < max_iterations; iter++){
		int ind_begin, ind_end, shift;
		/* Even iterations, scanline */
		if ( iter % 2 == 0 ){
			ind_begin = 0;
			ind_end   = target_ind_size;
			shift     = -1;
		/* Odd iterations, reverse scanline */
		}else{
			ind_begin = target_ind_size - 1;
			ind_end   = -1;
			shift     = 1;
		}

		/* loop through the target image */
		// #pragma omp for schedule(static) ordered
		#pragma omp parallel for schedule(static) default(shared)
		for (int count = 0; count<target_ind_size; count++){
			// srand(time(NULL)+omp_get_thread_num());

			int   targ_ind = target_ind[ind_begin-count*shift];
			float_type dist = distances[targ_ind];
			int   nn   = neighbors[targ_ind];

			/* Propagation step */
			/* Left\Right neighbor */
			int shifted_ind = targ_ind + shift;
			if ( shifted_ind>=0 && shifted_ind<target_mask_size && target_mask[shifted_ind]>0 ){
				int candidate_nn = neighbors[shifted_ind] - shift;

				if ( candidate_nn>=0 && candidate_nn<source_mask_size && source_mask[candidate_nn]>0 ){
					float_type candidate_dist = L2d2(candidate_nn, targ_ind);
					if ( candidate_dist < dist ){
						dist = candidate_dist;
						nn   = candidate_nn;
					}
				}
			}
			/* Above\Below neighbor */
			shifted_ind = targ_ind + shift * im_x;
			if ( shifted_ind>=0 && shifted_ind<target_mask_size && target_mask[shifted_ind]>0 ){
				int candidate_nn = neighbors[shifted_ind] - shift * im_x;

				if ( candidate_nn>=0 && candidate_nn<source_mask_size && source_mask[candidate_nn]>0 ){
					float_type candidate_dist = L2d2(candidate_nn, targ_ind);
					if ( candidate_dist < dist ){
						dist = candidate_dist;
						nn   = candidate_nn;
					}
				}
			}

			/* Random search step */
			unsigned long long int seed1 = time(NULL) + omp_get_thread_num();
			unsigned long long int seed2;
			int nn_x = nn % source_x;
			int nn_y = nn / source_x;
			int rand_shots = max_rand_shots;
			for (int w_size = window_size; w_size >= 1; w_size /= 2, rand_shots /= 2){
				rand_shots = rand_shots<3 ? 3 : rand_shots;

				/* truncate window to account for the source image size */
				int x_min = ((nn_x-w_size) > 0)    	   ? nn_x - w_size : 0;
				int y_min = ((nn_y-w_size) > 0)        ? nn_y - w_size : 0;
				int x_max = ((nn_x+w_size) < source_x) ? nn_x + w_size : source_x - 1;
				int y_max = ((nn_y+w_size) < source_y) ? nn_y + w_size : source_y - 1;

				/* sample max_rand_shots pixels from the window around current nn */
				int random_no_improve = 0;
				for (int k = 0; k < rand_shots; k++){
					// int candidate_nn = source_x * ( y_min + rand() % (y_max-y_min) ) + x_min + rand() % (x_max-x_min);
					seed2 = RAND(seed1); seed1 = RAND(seed2);
					// note that only lowest 32 bits are taken
					int candidate_nn = source_x * ( y_min + (seed1 & 0xFFFFFFFF) % (y_max-y_min) ) + x_min + (seed2 & 0xFFFFFFFF) % (x_max-x_min);

					if ( source_mask[candidate_nn] > 0 ){
						float_type candidate_dist = L2d2(candidate_nn, targ_ind);
						if ( candidate_dist < dist ){
							dist = candidate_dist;
							nn   = candidate_nn;
						}
						else{
							random_no_improve += 1;
						}
					}

					if (random_no_improve>10) break;
				}
			}

			if ( dist < distances[targ_ind] ){
				distances[targ_ind] = dist;
				neighbors[targ_ind] = nn;
			}
		}


		// /* Max distance */
		// max_dist_old = max_dist;
		// max_dist = distances[target_ind[ind_begin]];
		// for (int ind = ind_begin-shift; ind != ind_end; ind -= shift)
		// 	max_dist = fmaxf(max_dist,distances[target_ind[ind]]);
		// no_improve_iters = (fabsf(max_dist-max_dist_old)<TOL) ? no_improve_iters+1 : 0;


		/* Max distance */
		max_dist_old = max_dist;
		max_dist = distances[target_ind[ind_begin]];
		if ( iter % 2 == 0 ){
			#pragma omp parallel for schedule(static) reduction(max:max_dist) default(shared)
			for (int ind = ind_begin+1; ind<ind_end; ind++)
				max_dist = fmaxf(max_dist,distances[target_ind[ind]]);
		}else{
			#pragma omp parallel for schedule(static) reduction(max:max_dist) default(shared)
			for (int ind = ind_begin-1; ind>ind_end; ind--)
				max_dist = fmaxf(max_dist,distances[target_ind[ind]]);
		}
		no_improve_iters = (fabsf(max_dist-max_dist_old)<TOL) ? no_improve_iters+1 : 0;


		/* Early break if desired tolerance is achieved or there is no improvement */
		if ( no_improve_iters>6 || max_dist < TOL ) break;

	}
}