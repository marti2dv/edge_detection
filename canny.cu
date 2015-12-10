#include "utils.h"
#include <stdio.h>

#define PI 3.14159265

__global__  void convolution(unsigned char *inputChannel,
                         unsigned char *outputChannel, int numRows, int numCols,
                         float *filter, const int filterWidth)
{
    // Get 2D and 1D indexes
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int thread_idx = numCols * y_idx + x_idx;

    if ((x_idx < numCols) && (y_idx < numRows)) {
    	// get thread location in relation to the filter
    	// this will always give center of filter array width, since filterWidth must be odd to have center pixel
    	int center = (filterWidth - 1) / 2;

    	int x, y, idx, filter_idx;
    	float total = 0;
    	for (int i = 0; i < filterWidth; i++){
    	    for (int j = 0; j < filterWidth; j++){
        		x = x_idx + j - center; // x value of surrounding pixel
        		y = y_idx + i - center; // same for y value
                if (x < 0){ x = 0; } // clamping values outside image boundaries
                if (y < 0){ y = 0; }
                if (x >= numCols){ x = numCols - 1; }
                if (y >= numRows){ y = numRows -1; }
        		idx = numCols * y + x; // 1D index for surrounding pixel
        		filter_idx = filterWidth * i + j; // 1D index of weight value in filter
                float imageColor = inputChannel[idx];
                float filterColor = filter[filter_idx];
        		total += imageColor * filterColor;
    	    }
    	}
    	outputChannel[thread_idx] = total;
    }

}

__global__  void convolution(unsigned char *inputChannel,
                         float *outputChannel, int numRows, int numCols,
                         float *filter, const int filterWidth)
{
    // Get 2D and 1D indexes
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;
    int thread_idx = numCols * y_idx + x_idx;

    if ((x_idx < numCols) && (y_idx < numRows)) {
    	// get thread location in relation to the filter
    	// this will always give center of filter array width, since filterWidth must be odd to have center pixel
    	int center = (filterWidth - 1) / 2;

    	int x, y, idx, filter_idx;
    	float total = 0;
    	for (int i = 0; i < filterWidth; i++){
    	    for (int j = 0; j < filterWidth; j++){
        		x = x_idx + j - center; // x value of surrounding pixel
        		y = y_idx + i - center; // same for y value
                if (x < 0){ x = 0; } // clamping values outside image boundaries
                if (y < 0){ y = 0; }
                if (x >= numCols){ x = numCols - 1; }
                if (y >= numRows){ y = numRows -1; }
        		idx = numCols * y + x; // 1D index for surrounding pixel
        		filter_idx = filterWidth * i + j; // 1D index of weight value in filter
                float imageColor = inputChannel[idx];
                float filterColor = filter[filter_idx];
        		total += imageColor * filterColor;
    	    }
    	}
    	outputChannel[thread_idx] = total;
    }

}

__global__ void gradient_magnitude(float *d_inputChannel_x,
                                   float *d_inputChannel_y,
                                   unsigned char *outputChannel,
                                   size_t numRows, size_t numCols)
{
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx >= numCols || y_idx >= numRows){return;}

    int idx = numCols * y_idx + x_idx;
    float x = d_inputChannel_x[idx];
    float y = d_inputChannel_y[idx];
    // store in intermediate variable for precision
    float distance = sqrt((x*x) + (y*y));
    outputChannel[idx] = distance;
}

__global__ void gradient_direction(float *d_input_x,
                                   float *d_input_y,
                                   float *d_output,
                                   size_t numRows, size_t numCols)
{
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx >= numCols || y_idx >= numRows){return;}

    int idx = numCols * y_idx + x_idx;
    float x = d_input_x[idx];
    float y = d_input_y[idx];
    float angle = abs(atan2(y, x)) * 180 / PI;
    float direction;

    if (angle < 22.5 || angle > 157.5)
        direction = 0;
    else if (angle >= 22.5 && angle < 67.5)
        direction = 45;
    else if (angle >= 67.5 && angle < 112.5)
        direction = 90;
    else // angle between 112.5 and 157.5
        direction = 135;

    d_output[idx] = direction;
}

// perform gradient using specified operator (Sobel/Prewitt/Robert's Cross)
void gradient(unsigned char *d_image, float *d_directionImage, size_t numRows,
              size_t numCols, float *h_operator_x, float *h_operator_y,
              int operatorWidth, const dim3 block, const dim3 grid)
{
    const size_t numElems = numRows * numCols;
    const size_t mem_size = numElems * sizeof(float);
    const size_t filter_mem = operatorWidth * operatorWidth * sizeof(float);

    // allocate space for x and y gradients of input image
    float *d_Xgradient, *d_Ygradient;
    checkCudaErrors(cudaMalloc((void**) &d_Xgradient, mem_size));
    checkCudaErrors(cudaMalloc((void**) &d_Ygradient, mem_size));

    // alloate space for the operator kernel
    float *d_operator_x, *d_operator_y;
    checkCudaErrors(cudaMalloc((void**) &d_operator_x, filter_mem));
    checkCudaErrors(cudaMalloc((void**) &d_operator_y, filter_mem));
    cudaMemcpy(d_operator_x, h_operator_x, filter_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_operator_y, h_operator_y, filter_mem, cudaMemcpyHostToDevice);

    // calculate x gradient
    convolution<<<grid, block>>>(d_image, d_Xgradient, numRows, numCols,
                                   d_operator_x, operatorWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // calculate y gradient
    convolution<<<grid, block>>>(d_image, d_Ygradient, numRows, numCols,
                                   d_operator_y, operatorWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // calculate magnitude based on gradients
    gradient_magnitude<<<grid, block>>>(d_Xgradient, d_Ygradient, d_image,
                                        numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // populate d_directionImage with direction that image is pointing
    gradient_direction<<<grid, block>>>(d_Xgradient, d_Ygradient, d_directionImage,
                                        numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_Xgradient));
    checkCudaErrors(cudaFree(d_Ygradient));
    checkCudaErrors(cudaFree(d_operator_x));
    checkCudaErrors(cudaFree(d_operator_y));
}

// check pixels in front and behind of the direction that each pixel is facing
// any pixel that is not a local maximum is suppressed by being set to 0
__global__ void nonMaximumSuppression(unsigned char *d_magnitude, float *d_direction,
                                      size_t numRows, size_t numCols)
{
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx >= numCols || y_idx >= numRows){return;}

    int idx = numCols * y_idx + x_idx;

    // calculate neigbouring magnitudes
    int x1, y1, x2, y2;
    float m1 = -1; // if magnitude can't be set (neighbor is out of bounds),
    float m2 = -1; // -1 ensures out of bounds magnitude can't be local maxima
    if (d_direction[idx] == 0){
        x1 = x_idx + 1; y1 = y_idx;
        if (x1 < numCols){m1 = d_magnitude[numCols * y1 + x1];}
        x2 = x_idx - 1; y2 = y_idx;
        if (x2 >= 0){m2 = d_magnitude[numCols * y2 + x2];}
    }
    else if (d_direction[idx] == 45){
        x1 = x_idx + 1, y1 = y_idx + 1;
        if (x1 < numCols && y1 < numRows){m1 = d_magnitude[numCols * y1 + x1];}
        x2 = x_idx - 1, y2 = y_idx - 1;
        if (x2 >= 0 && y2 >= 0){m2 = d_magnitude[numCols * y2 + x2];}
    }
    else if (d_direction[idx] == 90){
        x1 = x_idx, y1 = y_idx + 1;
        if (y1 < numRows){m1 = d_magnitude[numCols * y1 + x1];}
        x2 = x_idx, y2 = y_idx - 1;
        if (y2 >= 0){m2 = d_magnitude[numCols * y2 + x2];}
    }
    else { // direction == 135
        x1 = x_idx - 1, y1 = y_idx + 1;
        if (x1 >= 0 && y1 < numRows){m1 = d_magnitude[numCols * y1 + x1];}
        x2 = x_idx + 1, y2 = y_idx - 1;
        if (x2 < numCols && y2 >= 0){m2 = d_magnitude[numCols * y2 + x2];}
    }

    if(d_magnitude[idx] < m1 || d_magnitude[idx] < m2)
        d_magnitude[idx] = 0;
}

__global__ void doubleThresholding(unsigned char *d_image, int high_threshold,
                                   int low_threshold, size_t numRows, size_t numCols)
{
    int x_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int y_idx = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_idx >= numCols || y_idx >= numRows){return;}

    int idx = numCols * y_idx + x_idx;

    if (d_image[idx] < low_threshold){
        d_image[idx] = 0;
    }
    else if(d_image[idx] >= low_threshold && d_image[idx] <= high_threshold){
        for (int y = -1; y <= 1; y++){
            if (y < numRows && y >= 0){
                for (int x = -1; x <= 1; x++){
                    if (x < numCols && x >= 0){
                        // if one neighbor is above the high_threshold, keep the edge
                        if (d_image[numCols * y + x] > high_threshold){return;}
                    }
                }
            }
        }
        // high threshold neighbor not found
        d_image[idx] = 0;
    }
}

void canny(unsigned char *d_inputImage, unsigned char *d_outputImage,
           const size_t numRows, const size_t numCols, float *h_filter,
           const int filterWidth, int high_threshold, int low_threshold,
           float *h_operator_x, float *h_operator_y, int operatorWidth)
{
    const int BLOCK_SIZE = 32;
    const size_t numElems = numRows * numCols;
    const size_t mem_size = numElems * sizeof(unsigned char);
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    const dim3 grid(numCols/BLOCK_SIZE + 1, numRows/BLOCK_SIZE + 1, 1);


    unsigned char *d_intermediateImage; // image we will be operating on
    checkCudaErrors(cudaMalloc((void**) &d_intermediateImage, mem_size));

    float *d_directionImage; // image storing line directions
    checkCudaErrors(cudaMalloc((void**) &d_directionImage, numElems * sizeof(float)));

    float *d_filter; // filter for image convolution
    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

    // step 1: gaussian blur/filter
    convolution<<<grid, block>>>(d_inputImage, d_intermediateImage, numRows, numCols,
                                 d_filter, filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // step 2: finding intensity gradient
    gradient(d_intermediateImage, d_directionImage, numRows, numCols,
             h_operator_x, h_operator_y, operatorWidth, block, grid);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // step 3: suppressing values that are not local maxima
    nonMaximumSuppression<<<grid, block>>>(d_intermediateImage, d_directionImage,
                                           numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // step 4: use double thresholding to eliminate values
    doubleThresholding<<<grid, block>>>(d_intermediateImage, high_threshold,
                                        low_threshold, numRows, numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    cudaMemcpy(d_outputImage, d_intermediateImage, mem_size, cudaMemcpyDeviceToDevice);

    checkCudaErrors(cudaFree(d_intermediateImage));
    checkCudaErrors(cudaFree(d_directionImage));
    checkCudaErrors(cudaFree(d_filter));
}
