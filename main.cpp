//Udacity HW2 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include "reference_calc.h"
#include "compare.h"

//include the definitions of the above functions for this homework
#include "edge_detection.cpp"


/*******  DEFINED IN student_func.cu *********/

void canny(unsigned char *d_inputImage, unsigned char *d_outputImage,
           const size_t numRows, const size_t numCols, float *h_filter,
           const int filterWidth, int high_threshold, int low_threshold,
           float *h_operator_x, float *h_operator_y, int operatorWidth);

/*******  Begin main *********/

int main(int argc, char **argv) {
  unsigned char *h_inputImage,  *d_inputImage;
  unsigned char *h_outputSobel, *h_outputPrewitt, *h_outputCross;
  unsigned char *d_outputSobel, *d_outputPrewitt, *d_outputCross;

  float *h_filter;
  int filterWidth;

  std::string input_file;
  std::string sobel_file = "output_sobel.png";
  std::string prewitt_file = "output_prewitt.png";
  std::string cross_file = "output_cross.png";

  if(argc != 4){
      printf("usage: ./edge_detection [input image] [high_threshold] [low_threshold]\n");
      return 1;
  }
  input_file = std::string(argv[1]);
  int high_threshold = std::atoi(argv[2]);
  int low_threshold = std::atoi(argv[3]);

  //load the image and give us our input and output pointers
  preProcess(&h_inputImage, &h_outputSobel, &h_outputPrewitt, &h_outputCross,
             &d_inputImage, &d_outputSobel, &d_outputPrewitt, &d_outputCross,
             &h_filter, &filterWidth, input_file);

  // set edge detection operators
  // operators are flipped in x and y directions for the convolution
  int operatorWidth = 3;
  int op_size = operatorWidth * operatorWidth;
  float h_sobel_x[op_size] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_sobel_y[op_size] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  float h_prewitt_x[op_size] = {1, 0, -1, 1, 0, -1, 1, 0, -1};
  float h_prewitt_y[op_size] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
  // cross operator must be padded to an odd width since it is a 2x2 kernel
  float h_cross_x[op_size] = {-1, 0, 0, 0, 1, 0, 0, 0, 0};
  float h_cross_y[op_size] = {0, -1, 0, 1, 0, 0, 0, 0, 0};

  GpuTimer timer;
  timer.Start();

  canny(d_inputImage, d_outputSobel, imageInput.rows, imageInput.cols, h_filter,
        filterWidth, high_threshold, low_threshold, h_sobel_x, h_sobel_y, operatorWidth);

  timer.Stop();
  printf("Sobel edge detection ran in: %f msecs.\n", timer.Elapsed());

  timer.Start();

  canny(d_inputImage, d_outputPrewitt, imageInput.rows, imageInput.cols, h_filter,
        filterWidth, high_threshold, low_threshold, h_prewitt_x, h_prewitt_y, operatorWidth);

  timer.Stop();
  printf("Prewitt edge detection ran in: %f msecs.\n", timer.Elapsed());

  timer.Start();

  canny(d_inputImage, d_outputCross, imageInput.rows, imageInput.cols, h_filter,
        filterWidth, high_threshold, low_threshold, h_cross_x, h_cross_y, operatorWidth);

  timer.Stop();

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Robert's cross edge detection ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the blurred image

  //copy the output back to the host
  size_t numPixels = imageInput.rows * imageInput.cols;
  checkCudaErrors(cudaMemcpy(h_outputSobel, d_outputSobel__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_outputPrewitt, d_outputPrewitt__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_outputCross, d_outputCross__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  postProcess(sobel_file, prewitt_file, cross_file);

  cleanUp();

  return 0;
}
