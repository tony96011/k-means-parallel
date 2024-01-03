#include "lab1_io.h"

#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <malloc.h>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>

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

__global__ void assign_clusters(Point *points, Point *centr,int K, int N, double *distances){
    
    //computing distance of a point and assigning all the data points, a centroid/cluster value
    int point_id = blockIdx.x*1024+threadIdx.x;
    if(point_id>N) return;

    for(int j=0; j<K ; j++){
        distances[point_id*K+j] = euclid(points[point_id], centr[j]);
    }
    int index = 0;
    for(int i = 1; i < K; i++)
    {
        if(distances[point_id*K+i] < distances[point_id*K+index])
            index = i;
    }
    //assigning the minimum distance cluster, which is an index
    points[point_id].cluster = index;
}

//funtion to recompute the new centroids for each cluster
//N is the total number of data points and K is the total number of clusters
__global__ void mean_recompute(int N, Point *points, Point *centr){
    int count = 0;
    Point sum;
    sum.x = 0;
    sum.y = 0;
    sum.z = 0;
    int cluster_id = threadIdx.x;
    for(int i=0; i< N ; i++){
        if(cluster_id == points[i].cluster){
            count++;
            sum.x += points[i].x;
            sum.y += points[i].y;
            sum.z += points[i].z;
        } 
    }
    centr[cluster_id].x = sum.x/count;
    centr[cluster_id].y = sum.y/count;
    centr[cluster_id].z = sum.z/count;
}

//driver function
void kmeans_CUDA(int N, int K, int* data_points, int** data_point_cluster, float** centroids, int* num_iterations){

    Point points[N];
    Point *d_points, *d_centr;
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
    cudaMalloc((void**)&d_points, N * sizeof(Point));
    cudaMemcpy(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_centr, K * sizeof(Point));
    cudaMemcpy(d_centr, centr, K * sizeof(Point), cudaMemcpyHostToDevice); 

    //---------------------------
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    
    // //array to keep a track of distances of a point from all centroids, to take the minimum out of them

    double *d_distances;
    cudaMalloc((void**)&d_distances, N * K * sizeof(double));
    //cudaMemcpy(d_distances, distances,N * K * sizeof(double), cudaMemcpyHostToDevice); 
    // //computing distance of a point and assigning all
    // for (int i=0; i<N; i++){
    //     for(int j=0; j<K ; j++){
    //         distances[j] = euclid(points[i], centr[j]);
    //     }
    //     int index = 0;
    //     for(int i = 1; i < K; i++){
    //         if(distances[i] < distances[index])
    //             index = i;
    //     }
    //     points[i].cluster = index;
    // }
    assign_clusters<<<blocks, threads>>>(d_points, d_centr, K, N, d_distances);
    cudaDeviceSynchronize();

    //---------------------------

    mean_recompute<<<1, K>>>(N, d_points, d_centr);
    cudaDeviceSynchronize();
    // putback(centr, K);

    //---------------------------

    int iterations = 1;
    int count;
    do {
        // mean_recompute(K, N, points,centr);
        // putback(centr, K);
        mean_recompute<<<1, K>>>(N, d_points, d_centr);
        cudaDeviceSynchronize();
        //storing old values for convergence check
        int old[N];
        for (int i=0; i<N; i++){
            old[i] = points[i].cluster;
        }
        // assignclusters(points, centr, K, N);
        assign_clusters<<<blocks, threads>>>(d_points, d_centr, K, N, d_distances);
        cudaDeviceSynchronize();
        cudaMemcpy(points, d_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
        iterations++;
        count = 0;
        for (int i=0; i<N; i++){
            if (old[i] == points[i].cluster)
                count++;
        }
    } while(count!=N);
    printf("%d\n", iterations);

    cudaMemcpy(points, d_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(centr, d_centr, K * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaFree(d_points);
    cudaFree(d_centr);
    //---------------------------

    *data_point_cluster= (int*) calloc(4*N, sizeof(int));
    *centroids = (float*) calloc(vect.size(), sizeof(float));

    int q = 0;
    for (int i = 0; i< 4*N; i+=4){
        data_point_cluster[0][i] = points[q].x;
        data_point_cluster[0][i+1] = points[q].y;
        data_point_cluster[0][i+2] = points[q].z;
        data_point_cluster[0][i+3] = points[q].cluster;
        q++;
    }
    for (int i = 0; i<vect.size(); i++){
        centroids[0][i] = vect[i];
    }

    * num_iterations = vect.size()/K -1 ;
}

//funtion to recompute the new centroids for each cluster
//N is the total number of data points and K is the total number of clusters
// void mean_recompute(int K, int N, Point points[], Point centr[]){
//     int count[K];
//     Point sum[K];
//     for(int i=0; i< N ; i++){
//         count[points[i].cluster]++;
//         sum[points[i].cluster] = addtwo(points[i],sum[points[i].cluster] );
//     }
//     for(int i=0; i< K ; i++){
//         centr[i].x = sum[i].x/count[i];
//         centr[i].y = sum[i].y/count[i];
//         centr[i].z = sum[i].z/count[i];
//     }
// }

//assuming they are the same cluster
__device__ Point addtwo(Point a, Point b){
    Point ans;
    ans.x = a.x + b.x;
    ans.y = a.y + b.y;
    ans.z = a.z + b.z;
    ans.cluster = a.cluster;
    return ans;
}

// void assignclusters(Point points[], Point centr[],int K, int N){
//     double distances[K];
//     //computing distance of a point and assigning all the data points, a centroid/cluster value
//     for (int i=0; i<N; i++)
//     {
//         for(int j=0; j<K ; j++){
//             distances[j] = euclid(points[i], centr[j]);
//         }
//         int index = 0;
//         for(int i = 1; i < K; i++)
//         {
//             if(distances[i] < distances[index])
//                 index = i;
//         }
//         //assigning the minimum distance cluster, which is an index
//         points[i].cluster = index;
//     }
// }

//function to calculate euclidea distance between two points
__device__ double euclid(Point a, Point b){
    double x = a.x- b.x;
    double y = a.y- b.y;
    double z = a.z- b.z;
    double dist = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
    return dist;
}

// __device__ void putback(Point centr[],int K){
//     for (int i =0; i<K; i++) {
//         vect.push_back(centr[i].x);
//         vect.push_back(centr[i].y);
//         vect.push_back(centr[i].z);
//     }
// }

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
	int* data_points;		//data points (input)
	int* cluster_points;	//clustered data points (to be computed)
	float* centroids;			//centroids of each iteration (to be computed)
	int num_iterations;    //no of iterations performed by algo (to be computed)
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
	centroids_out (argv[4], K, num_iterations, centroids);

	computation_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
	printf("Time Taken: %lf \n", computation_time);
	
	return 0;
}