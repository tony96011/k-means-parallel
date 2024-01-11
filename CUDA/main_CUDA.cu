#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <malloc.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#define double float

using namespace std;

vector<int> vect;

struct Point{
    double x; //x coordinate
    double y; //y coordinate
    double z; //z coordinate
    int cluster; //cluster to which the point belongs
};

//helper function prototypes
__device__ Point addtwo(Point a, Point b);
__device__ double euclid(Point a, Point b);
__device__ Point mean(Point arr[], int N);
// __device__ void putback(Point centr[],int K);

__global__ void assign_clusters(Point *points, Point *centr,int K, int N, double *distances, short *d_modify_record){
    /*share mem*/
    // __shared__ Point s_centr[10];
    // int thread_id = threadIdx.x;
    // if(thread_id < K){
    //     s_centr[thread_id] = centr[thread_id];
    // }
    // __syncthreads();

    //computing distance of a point and assigning all the data points, a centroid/cluster value
    int point_id = blockIdx.x*1024+threadIdx.x;
    Point cur = points[point_id];
    if(point_id>=N) return;

    #pragma unroll
    for(int j=0; j<K ; j++){
        distances[point_id*K+j] = euclid(cur, centr[j]);
    }
    int index = 0;
    
    #pragma unroll
    for(int i = 1; i < K; i++)
    {
        if(distances[point_id*K+i] < distances[point_id*K+index])
            index = i;
    }
    //assign modify to 1 if same and 0 if different
    d_modify_record[point_id] = (cur.cluster ^ index) != 0;
    //assigning the minimum distance cluster, which is an index
    points[point_id].cluster = index;
}

__global__ void update_centroids(int K, Point *sum, int *count, Point *centr){
    int cluster_id = threadIdx.x;
    int count_cluster = count[cluster_id];
    Point cur_sum = sum[cluster_id];
    centr[cluster_id].x = cur_sum.x/count_cluster;
    centr[cluster_id].y = cur_sum.y/count_cluster;
    centr[cluster_id].z = cur_sum.z/count_cluster;
}
__device__ float atomicAddDouble(float* address, float val) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val + __uint_as_float(assumed)));
    } while (assumed != old);

    return __uint_as_float(old);
}


// Sum all data points into sum and count the number of points in the specific cluster with "count"
__global__ void Points_Sum_Up(int N, int K, Point *points, Point* sum, int* count){
    extern __shared__ Point s_sum[];
    int point_id = blockIdx.x*1024+threadIdx.x;
    int thread_id = threadIdx.x;
    if(point_id>=N) return;

    if(threadIdx.x < K){
        s_sum[threadIdx.x].x = 0;
        s_sum[threadIdx.x].y = 0;
        s_sum[threadIdx.x].z = 0;
    }
    __syncthreads();

    Point cur = points[point_id];
    int cur_cluster = cur.cluster;
    atomicAdd(&count[cur_cluster], 1);
    /*TODO:should use Atomic but there's no atomic add for double in current GPU board*/
    atomicAddDouble(&s_sum[cur_cluster].x, cur.x);
    atomicAddDouble(&s_sum[cur_cluster].y, cur.y);
    atomicAddDouble(&s_sum[cur_cluster].z, cur.z);

    __syncthreads();

    if (thread_id < K) {
        //printf("%f\n", s_sum[thread_id].x);
        atomicAddDouble(&sum[thread_id].x, s_sum[thread_id].x);
        atomicAddDouble(&sum[thread_id].y, s_sum[thread_id].y);
        atomicAddDouble(&sum[thread_id].z, s_sum[thread_id].z);
    }
}

__global__ void check_modify(short *modify_record, int *d_not_done, int N){
    int point_id = blockIdx.x*1024+threadIdx.x;
    if(point_id>N) return;
    if(modify_record[point_id] == 1){
        atomicOr(d_not_done, 1);
    }
}

__global__ void clear(Point *sum, int *count){
    int cluster_id = threadIdx.x;
    sum[cluster_id].x = 0;
    sum[cluster_id].y = 0;
    sum[cluster_id].z = 0;
    count[cluster_id] = 0;
}

//driver function
Point* points;
void kmeans_CUDA(int N, int K, int* data_points, int** data_point_cluster, float** centroids, int* num_iterations){

    int* d_not_done, *d_count;
    int not_done;
    points = (Point*) malloc(N * sizeof(Point));
    //array to keep a track of distances of a point from all centroids, to take the minimum out of them
    double *d_distances;
    short *d_modify_record;
    Point *d_points, *d_centr,*d_sum;
    //---------------------------
    
    int j=0;
    for (int i=0;  i<(3*N); i+=3){
        points[j].x  = data_points[i];
        points[j].y  = data_points[i+1];
        points[j].z  = data_points[i+2];
        j++;
    }

    //---------------------------

    //random centroid initialization. centroids are random points
    //center is the array of centroids containing their locations and cluster number as their index in this array
    srand(10);
    Point centr[K];
    for (int i = 0; i< K; i++){
        int random = rand()%N;//some random value
        centr[i] = points[random];
    }

    cudaMalloc((void**)&d_distances, N * K * sizeof(double));
    cudaMalloc((void**)&d_points, N * sizeof(Point));
    cudaMemcpyAsync(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_centr, K * sizeof(Point));
    cudaMemcpyAsync(d_centr, centr, K * sizeof(Point), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&d_modify_record, N * sizeof(short));
    cudaMalloc((void**)&d_not_done, sizeof(int));
    cudaMemcpyAsync(d_not_done, &not_done, sizeof(int), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&d_sum, K * sizeof(Point));
    cudaMalloc((void**)&d_count, K * sizeof(int));

    //---------------------------
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    
    assign_clusters<<<blocks, threads>>>(d_points, d_centr, K, N, d_distances, d_modify_record);
    cudaDeviceSynchronize();

    int iterations = 1;

    do {
        /*initial not_done*/
        not_done = 0;
        cudaMemcpy(d_not_done, &not_done, sizeof(int), cudaMemcpyHostToDevice);

        Points_Sum_Up<<<blocks, threads,K*sizeof(Point)>>>(N, K, d_points, d_sum, d_count);
        cudaDeviceSynchronize();
        update_centroids<<<1, K>>>(K, d_sum, d_count, d_centr);
        cudaDeviceSynchronize();
        clear<<<1, K>>>(d_sum, d_count);
        cudaDeviceSynchronize();

        assign_clusters<<<blocks, threads>>>(d_points, d_centr, K, N, d_distances, d_modify_record);
        cudaDeviceSynchronize();

        check_modify<<<blocks, threads>>>(d_modify_record, d_not_done, N);
        cudaDeviceSynchronize();
        cudaMemcpy(&not_done, d_not_done, sizeof(int), cudaMemcpyDeviceToHost);

        iterations++;
    } while(not_done);

    // printf("%d\n", iterations);

    cudaMemcpy(points, d_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(centr, d_centr, K * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_centr);
    cudaFree(d_distances);
    cudaFree(d_modify_record);
    cudaFree(d_not_done);
    cudaFree(d_sum);
    cudaFree(d_count);
    //---------------------------

    *data_point_cluster= (int*) calloc(4*N, sizeof(int));
    // *centroids = (float*) calloc(vect.size(), sizeof(float));

    int q = 0;
    for (int i = 0; i< 4*N; i+=4){
        data_point_cluster[0][i] = points[q].x;
        data_point_cluster[0][i+1] = points[q].y;
        data_point_cluster[0][i+2] = points[q].z;
        data_point_cluster[0][i+3] = points[q].cluster;
        q++;
    }
    // for (int i = 0; i<vect.size(); i++){
    //     centroids[0][i] = vect[i];
    // }

    * num_iterations = vect.size()/K -1 ;
}

//assuming they are the same cluster
__device__ Point addtwo(Point a, Point b){
    Point ans;
    ans.x = a.x + b.x;
    ans.y = a.y + b.y;
    ans.z = a.z + b.z;
    ans.cluster = a.cluster;
    return ans;
}

__device__ double fastPower(double base, int exponent) {

    double result = 1.0;
    double currentPower = base;

    // Use binary exponentiation
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= currentPower;
        }
        currentPower *= currentPower;
        exponent /= 2;
    }

    return result;
}

//function to calculate euclidea distance between two points
__device__ double euclid(Point a, Point b){
    double x = a.x- b.x;
    double y = a.y- b.y;
    double z = a.z- b.z;
    return fastPower(x, 2) + fastPower(y, 2) + fastPower(z, 2);
}

void dataset_in (const char* dataset_filename, int* N, int** data_points){
	FILE *fin = fopen(dataset_filename, "r");

	fscanf(fin, "%d", N);
	
	*data_points = (int*)malloc(sizeof(int)*((*N)*3));
	
	for (int i = 0; i < (*N)*3; i++){
		fscanf(fin, "%d", (*data_points + i));
	}

	fclose(fin);
}

void clusters_out (const char* cluster_filename, int N, int* cluster_points){
	FILE *fout = fopen(cluster_filename, "w");

	for (int i = 0; i < N; i++){
		fprintf(fout, "%d %d %d %d\n", 
			*(cluster_points+(i*4)), *(cluster_points+(i*4)+1), 
			*(cluster_points+(i*4)+2), *(cluster_points+(i*4)+3));
	}

	fclose(fout);
}

void centroids_out (const char* centroid_filename, int K, int num_iterations, float* centroids){
	FILE *fout = fopen(centroid_filename, "w");

	for (int i = 0; i < num_iterations+1; i++){				//ith iteration
		for (int j = 0; j < K; j++){			//jth centroid of ith iteration
			fprintf(fout, "%f %f %f, ", 
									*(centroids+(i*K+j)*3), 	 //x coordinate
									*(centroids+(i*K+j)*3+1),  //y coordinate
									*(centroids+(i*K+j)*3+2)); //z coordinate
		}
		fprintf(fout, "\n");
	}

	fclose(fout);
}


/*
	Arguments:
		arg1: K (no of clusters)
		arg2: input filename (data points)
		arg3: output filename (data points & cluster)
		arg4: output filename (centroids of each iteration)
*/

int main(int argc, char const *argv[])
{
	if (argc < 5){
		printf("\nLess Arguments\n");
		return 0;
	}

	if (argc > 5){
		printf("\nToo many Arguments\n");
		return 0;
	}

	//---------------------------------------------------------------------
	int N;					//no. of data points (input)
	int K;					//no. of clusters to be formed (input)
	int* cluster_points;	//clustered data points (to be computed)
	float* centroids;			//centroids of each iteration (to be computed)
	int num_iterations;    //no of iterations performed by algo (to be computed)
    int* data_points;		//data points (input)
	//---------------------------------------------------------------------

	clock_t start_time, end_time;
	double computation_time;

	K = atoi(argv[1]);

	/*
		-- Pre-defined function --
		reads dataset points from input file and creates array
		containing data points, using 3 index per data point, ie
	     -----------------------------------------------
		| pt1_x | pt1_y | pt1_z | pt2_x | pt2_y | pt2_z | ...
		 -----------------------------------------------
	*/
    
	dataset_in (argv[2], &N, &data_points);
	start_time = clock();
	// /*
	// 	*****************************************************
	// 		TODO -- You must implement this function
	// 	*****************************************************
	// */
    
	kmeans_CUDA(N, K, data_points, &cluster_points, &centroids, &num_iterations);
	end_time = clock();

	// /*
	// 	-- Pre-defined function --
	// 	reads cluster_points and centroids and save it it appropriate files
	// */
    
	clusters_out (argv[3], N, cluster_points);
	// centroids_out (argv[4], K, num_iterations, centroids);

	computation_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("Time Taken: %lf \n", computation_time);
	
	return 0;
}
