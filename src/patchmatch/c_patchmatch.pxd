cdef extern from "c_patchmatch.h":
	void pm_64( double* source, char* source_mask, int* source_ind, int source_y, int source_x, int source_ch,
			 double* target, char* target_mask, int* target_ind, int target_y, int target_x, int target_ch, int target_ind_size,
			 int *neighbors, double *distances, int* patch_ind, int patch_size, double* weight, double* lambdas, int max_rand_shots, int max_iterations, int max_window_size, double TOL );
	void pm( float* source, char* source_mask, int* source_ind, int source_y, int source_x, int source_ch,
			 float* target, char* target_mask, int* target_ind, int target_y, int target_x, int target_ch, int target_ind_size,
			 int *neighbors, float *distances, int* patch_ind, int patch_size, float* weight, float* lambdas, int max_rand_shots, int max_iterations, int max_window_size, float TOL );

# "Cython style" .h declaration file to be imported in .pyx files with cimport