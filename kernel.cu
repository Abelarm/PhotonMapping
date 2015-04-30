/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/
#include <iostream>
#include <fstream>
#include <dos.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_profiler_api.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <cstdlib>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <iostream>
#include <fstream>

#define SPHERES 5
#define DIM 1200
#define DIMTH 768
#define NUMTEST 200

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 1024
#define LIGHTX 380
#define LIGHTY 768
#define LIGHTZ 20
#define NUMREFLECTION 3

using namespace std;
__device__ double pow(double x,double y){

return x*x;
}



__device__ int controlSide(int indice,int *matrix, int radiusAVG){
	int countAVG = 0;
	if(indice + radiusAVG <= DIM * DIM && indice - radiusAVG >= 0){
			//controllo i lati della posizione
			
			for(int i = 1; i < radiusAVG; i++){

				int dx = i;
				int sx = -i;

				countAVG+=matrix[indice + dx];
				
				countAVG+=matrix[indice + sx];
				


			}

		}


	return countAVG;
}

struct Sphere {
	float   r, b, g;
	float   radius;
	float   x, y, z;
	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			//*n = 1.0;
			return dz + z;
		
		}
		return -INF;
		
	}
};


struct Point{
	int x;
	int y;
	int z;
	double d;
	int j;

};

__device__ double dot3(double a[], double b[]){

	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}



__device__ void reflection(Point maxp, double dirx, double diry,double dirz, Sphere * s,unsigned char *ptr, int *matrix, Point *pointmatrix){
	
	int indice;
	Point points[SPHERES];

	for(int k=0; k<NUMREFLECTION;k++){

			int indx = maxp.j;

			double nX = maxp.x -  s[indx].x;
			double nY = maxp.y -  s[indx].y;
			double nZ = maxp.z -  s[indx].z;

			double a[] = {nX,nY,nZ};
			double b[] = {dirx,diry,dirz};

			float fact = 2*(dot3(a,b));
			double c[] = {fact*a[0],fact*a[1],fact*a[2]};
			double ray[] = {c[0] - b[0],c[1]-b[1],c[2]-b[2]};

			dirx=ray[0];
			diry=ray[1];
			dirz=ray[2];

		for (int j = 0; j<SPHERES; j++) {
			
				
				Point p;
				p.x=-1;
				p.y=-1;
				p.z=-1;
				p.d=-1;
				p.j=-1;

				//printf("Intersezione con sfera %d punti x=%f , y=%f z=%f\n",j,dirx,diry,dirz);

				float A = pow(dirx,2.0)+pow(diry,2.0)+pow(dirz,2.0);
				float B = 2.0 * (((LIGHTX-s[j].x)*dirx + (LIGHTY-s[j].y)*diry + (LIGHTZ - s[j].z )*dirz));
				float C = (pow((LIGHTX-s[j].x),2) + pow((LIGHTY-s[j].y),2) + pow((LIGHTZ-s[j].z),2)) - pow(s[j].radius,2);
				float D = B*B-4*A*C;
				

				if(D > 0.0){

					double t0 = ((-B) + sqrtf(D))/(2*A);
					double t1 = ((-B) - sqrtf(D))/(2*A);

					double ix0 = LIGHTX + (t0*dirx);
					double iy0 = LIGHTY + (t0*diry);
					double iz0 = LIGHTZ + (t0*dirz);

					double ix1 = LIGHTX + (t1*dirx);
					double iy1 = LIGHTY + (t1*diry);
					double iz1 = LIGHTZ + (t1*dirz);					

					double d0 = sqrt(pow((ix0-LIGHTX),2)+pow((iy0-LIGHTY),2)+pow((iz0-LIGHTZ),2));
					double d1 = sqrt(pow((ix1-LIGHTX),2)+pow((iy1-LIGHTY),2)+pow((iz1-LIGHTZ),2));

					int ix;
					int iy;
					int iz;
					double d;

					if(t0>0 && t1>0){
						if(t0<t1){
							ix = ix0;
							iy = iy0;
							iz = iz0;
							d=t0;
						}else{
							ix = ix1;
							iy = iy1;
							iz = iz1;
							d=t1;
						}
					}else{
						if(t0<0 && t1>0){
							ix = ix1;
							iy = iy1;
							iz = iz1;
							d=t1;
						}else{
							if(t1<0 && t0>0){
								ix = ix0;
								iy = iy0;
								iz = iz0;
								d=t0;
							}
						}
					}

					if(ix<0 || iy<0 || iz <0 || ix>DIM || iy>DIM || iz > DIM)
						continue;
				
					//hostmatrix[iy][ix]=1;

					p.x=ix;
					p.y=iy;
					p.z=iz;
					p.d=d;
					p.j=j;
					points[j]=p;

					
					//count++;
					

				}
			
				if(D < 0.0){
					points[j]=p;
				}

			}

			maxp.d=4294967295;

			for (int j = 0; j<SPHERES; j++) {


				if(points[j].x == -1 || (points[j].x == 0 && points[j].y == 0 && points[j].z == 0 && points[j].d == 0) || points[j].d<0 )
					continue;			

				if(maxp.d>points[j].d){
					maxp=points[j];
				}

			}

			if(maxp.d==4294967295)
				continue;

			indice = (DIM * maxp.y)+maxp.x;
			atomicAdd(&matrix[indice], (matrix[indice] +=1 * (NUMREFLECTION - k +1)));
			pointmatrix[indice] = maxp;
			
		}//end for(int k=0; k<NUMREFLECTION;k++)


}//fine metodo reflection





__global__ void setup_kernel( curandState *state, unsigned long seed)
{
    int id = threadIdx.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	seed = seed + offset;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void kernel(Sphere *s, unsigned char *ptr, int *matrix, curandState* globalState, Point *pointmatrix) {

	
	Point points[SPHERES];

	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = (DIM * y)+x;

	int ind = threadIdx.x;
    curandState localState = globalState[ind];
	int start = 0;
	int end = DIM;
	
	int count=0;


	for(int i=0;i<1;i++){
		
		float rnd_number = curand_uniform(&localState);
		int rnd_integer_from_A_to_B_X = start + rnd_number * (end-start);
		rnd_number = rnd_number = curand_uniform(&localState);
		int rnd_integer_from_A_to_B_Y = start + rnd_number * (end-start);
		rnd_number = rnd_number = curand_uniform(&localState);
		int rnd_integer_from_A_to_B_Z = start + rnd_number * (end-start);
		globalState[ind] = localState; 
		int RX = rnd_integer_from_A_to_B_X;
		int RY = rnd_integer_from_A_to_B_Y;
		int RZ = rnd_integer_from_A_to_B_Z;

		double dirx;
		double diry;
		double dirz;

		dirx=RX-LIGHTX;
		diry=RY-LIGHTY;
		dirz=RZ-LIGHTZ;

		for (int j = 0; j<SPHERES; j++) {
			
				
				Point p;
				p.x=-1;
				p.y=-1;
				p.z=-1;
				p.d=-1;
				p.j=-1;

				float A = pow(dirx,2.0)+pow(diry,2.0)+pow(dirz,2.0);
				float B = 2.0 * (((LIGHTX-s[j].x)*dirx + (LIGHTY-s[j].y)*diry + (LIGHTZ - s[j].z )*dirz));
				float C = (pow((LIGHTX-s[j].x),2) + pow((LIGHTY-s[j].y),2) + pow((LIGHTZ-s[j].z),2)) - pow(s[j].radius,2);
				float D = B*B-4*A*C;
				

				if(D > 0.0){
					

					double t0 = ((-B) + sqrtf(D))/(2*A);
					double t1 = ((-B) - sqrtf(D))/(2*A);

					double ix0 = LIGHTX + (t0*dirx);
					double iy0 = LIGHTY + (t0*diry);
					double iz0 = LIGHTZ + (t0*dirz);

					double ix1 = LIGHTX + (t1*dirx);
					double iy1 = LIGHTY + (t1*diry);
					double iz1 = LIGHTZ + (t1*dirz);					

					double d0 = sqrt(pow((ix0-LIGHTX),2)+pow((iy0-LIGHTY),2)+pow((iz0-LIGHTZ),2));
					double d1 = sqrt(pow((ix1-LIGHTX),2)+pow((iy1-LIGHTY),2)+pow((iz1-LIGHTZ),2));

					int ix;
					int iy;
					int iz;
					double d;

					if(t0>0 && t1>0){
						if(t0<t1){
							ix = ix0;
							iy = iy0;
							iz = iz0;
							d=t0;
						}else{
							ix = ix1;
							iy = iy1;
							iz = iz1;
							d=t1;
						}
					}else{
						if(t0<0 && t1>0){
							ix = ix1;
							iy = iy1;
							iz = iz1;
							d=t1;
						}else{
							if(t1<0 && t0>0){
								ix = ix0;
								iy = iy0;
								iz = iz0;
								d=t0;
							}
						}
					}

					if(ix<0 || iy<0 || iz <0 || ix>DIM || iy>DIM || iz > DIM)
						continue;

					p.x=ix;
					p.y=iy;
					p.z=iz;
					p.d=d;
					p.j=j;
					points[j]=p;


					count++;
					

				}
				if(D < 0.0){
					points[j]=p;
				}

			}

			Point maxp;
			maxp.d=4294967295;

			for (int j = 0; j<SPHERES; j++) {


				if(points[j].x == -1 || (points[j].x == 0 && points[j].y == 0 && points[j].z == 0 && points[j].d == 0) || points[j].d<0 )
					continue;

				

				if(maxp.d>points[j].d){
					maxp=points[j];
				}

			}

			if(maxp.d==4294967295)
				continue;

			int indice = (DIM * maxp.y)+maxp.x;
			atomicAdd(&matrix[indice],(matrix[indice] +=1 * NUMREFLECTION));
			pointmatrix[indice] = maxp;
			
		///////////////////RIMBALZI

			reflection( maxp, dirx,  diry, dirz,  s,ptr,matrix, pointmatrix);



			
		__syncthreads();


					int countAVG = 0;
					int radiusAVG = 2;
					
					countAVG += controlSide(indice,matrix,radiusAVG);
				
					for(int k = 1; k <= radiusAVG;k++){

						if( ( indice - (DIM * radiusAVG)) >= 0 && (indice + (DIM * radiusAVG) ) <= DIM*DIM){
						//controllo sopra e sotto la matrice
			
						for(int i = 1; i < radiusAVG; i++){

							int up = i * DIM;
							int down = -i * DIM;

							 countAVG += matrix[indice + up];
							
							countAVG  += matrix[indice + down];
							


							countAVG += controlSide(indice + up,matrix,radiusAVG);
				
							countAVG += controlSide(indice + down,matrix,radiusAVG);

						}


						}
		
				}
			//countAVG += matrix[indice];// conto il pixel centrale
			int centro = matrix[indice];
			Point center = pointmatrix[indice];
			Sphere mysphere = s[center.j];


			float media = countAVG/((radiusAVG*2+1)*(radiusAVG*2+1));
			float scale = 1-(NUMREFLECTION/media)+0.4;
			if(scale<0){
				scale = 0.4;
			}
			if(scale>1){
				scale=1;
			}
			
			ptr[indice * 4 + 0] = scale * (mysphere.r);
			ptr[indice * 4 + 1] = scale * (mysphere.g);
			ptr[indice * 4 + 2] = scale * (mysphere.b);
			ptr[indice * 4 + 3] = scale * 255;
			


	}

}//fine metodo kernel







// globals needed by the update routine
struct DataBlock {
	unsigned char   *dev_bitmap;
	Sphere          *s;
};

//singola esecuzione del Photon Mapping
float PhotonMap(void){
	cudaProfilerStart();
	DataBlock   data;
	CPUBitmap bitmap(DIM, DIM, &data);
	// capture the start time
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	unsigned char   *dev_bitmap;
	Sphere          *s;



	// allocate memory on the GPU for the output bitmap
	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

	unsigned char* temp_bitmap= (unsigned char*)malloc(bitmap.image_size());

	for(int i=0;i<DIM*DIM;i++){
			temp_bitmap[i * 4 + 0] = 0;
			temp_bitmap[i * 4 + 1] = 0;
			temp_bitmap[i * 4 + 2] = 0;
			temp_bitmap[i * 4 + 3] = 255;
	}

	cudaMemcpy(dev_bitmap, temp_bitmap, bitmap.image_size(), cudaMemcpyHostToDevice);

// allocate memory for the Sphere dataset
	cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES);

	int* hostmatrix;
	int* devicematrix;

	

	cudaMalloc((void**)&devicematrix, DIM*DIM*sizeof(int));
	hostmatrix = (int* )malloc(DIM*DIM*sizeof(int));

	for(int i = 0; i< DIM*DIM;i++){
	hostmatrix[i] = 0;
	}

	cudaMemcpy(devicematrix ,hostmatrix, sizeof(int) * DIM * DIM, cudaMemcpyHostToDevice);

	Point* pointmatrix, * fakematrix;
	cudaMalloc((void**)&pointmatrix, DIM*DIM*sizeof(Point));
	fakematrix = (Point* )malloc(DIM*DIM*sizeof(Point));
	Point p;
	p.x=-1;
	p.y=-1;
	p.z=-1;
	p.d=-1;
	p.j=-1;

	for(int i = 0; i< DIM*DIM;i++){
		fakematrix[i] = p;
	}

	cudaMemcpy(pointmatrix ,fakematrix, sizeof(Point) * DIM * DIM, cudaMemcpyHostToDevice);
	free(fakematrix);

	// allocate temp memory, initialize it, copy to
	// memory on the GPU, then free our temp memory
	srand(time(NULL));

	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i<SPHERES; i++) {
		
		temp_s[i].r = rand()%256;
		temp_s[i].g = rand()%256;
		temp_s[i].b = rand()%256;

		temp_s[i].radius = 30+(rand()%41);
		temp_s[i].x = rand()%DIM;
		temp_s[i].y =rand()%DIM;
		temp_s[i].z = rand()%DIM/6;
		//printf("Sfera %d : x=%f y=%f z=%f radius=%f\n", i, temp_s[i].x,temp_s[i].y,temp_s[i].z,temp_s[i].radius);
	}

	cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);
	

	// generate a bitmap from our sphere data
	dim3    grids(DIMTH / 16, DIMTH / 16);
	dim3    threads(16, 16);

	int N = DIMTH*DIMTH;
	curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );
    
    // setup seeds
    setup_kernel <<< grids, threads >>> ( devStates, time(NULL) );

	kernel << <grids, threads >> >(s, dev_bitmap, devicematrix,devStates,pointmatrix);
	
	cudaProfilerStop();



	// copy our bitmap back from the GPU for display
	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostmatrix, devicematrix, DIM*DIM*sizeof(int), cudaMemcpyDeviceToHost);

	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float   elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start, stop);
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	free(temp_s);
	cudaFree(dev_bitmap);
	cudaFree(s);
	cudaFree(devicematrix);
	cudaFree(pointmatrix);
	cudaFree(devStates);

	//bitmap.display_and_exit();
	return elapsedTime;
}

int main(int argc, char* argv[]) {

	ofstream myfile;
	char buffer[100];
	float realTime;
	float expectedTime=0.0;

	std::string name="PhotonMapping_S";
	name=name+itoa(SPHERES,buffer,10);
	name=name+"_D"+itoa(DIM,buffer,10);
	name=name+"_TH"+itoa(DIMTH,buffer,10);
	name=name+"_T"+itoa(NUMTEST,buffer,10)+".txt";
	myfile.open(name);

	for(int i=0;i<NUMTEST;i++){
		realTime=PhotonMap();
		expectedTime+=realTime;
		sprintf(buffer, "Experiment %d Time:  %3.1f ms\n", i+1, realTime);
		myfile<<buffer;
	}
	expectedTime=expectedTime/NUMTEST;
	//printf("Expected Time:  %3.1f ms\n", expectedTime);

	
	sprintf(buffer, "Mean Time:  %3.1f ms\n", expectedTime);
	myfile<<buffer;
	myfile.close();
}
