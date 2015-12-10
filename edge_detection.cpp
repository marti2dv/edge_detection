#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageInput;
cv::Mat imageSobel;
cv::Mat imagePrewitt;
cv::Mat imageCross;

unsigned char *d_inputImage__;
unsigned char *d_outputSobel__;
unsigned char *d_outputPrewitt__;
unsigned char *d_outputCross__;

float *h_filter__;


//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(unsigned char **h_inputImage, unsigned char **h_outputSobel,
                unsigned char **h_outputPrewitt, unsigned char **h_outputCross,
                unsigned char **d_inputImage, unsigned char **d_outputSobel,
                unsigned char **d_outputPrewitt, unsigned char **d_outputCross,
                float **h_filter, int *filterWidth,
                const std::string &filename) {

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  imageInput = cv::imread(filename.c_str(), 0);
  if (imageInput.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  //allocate memory for the output
  imageSobel.create(imageInput.rows, imageInput.cols, CV_8UC1);
  imagePrewitt.create(imageInput.rows, imageInput.cols, CV_8UC1);
  imageCross.create(imageInput.rows, imageInput.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInput.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *h_inputImage = imageInput.ptr<unsigned char>(0);
  *h_outputSobel = imageSobel.ptr<unsigned char>(0);
  *h_outputPrewitt = imagePrewitt.ptr<unsigned char>(0);
  *h_outputCross = imageCross.ptr<unsigned char>(0);

  const size_t numPixels = imageInput.rows * imageInput.cols;
  const int pixel_mem = sizeof(unsigned char) * numPixels;
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_inputImage, pixel_mem));

  checkCudaErrors(cudaMalloc(d_outputSobel, pixel_mem));
  checkCudaErrors(cudaMemset(*d_outputSobel, 0, pixel_mem)); //make sure no memory is left laying around

  checkCudaErrors(cudaMalloc(d_outputPrewitt, pixel_mem));
  checkCudaErrors(cudaMemset(*d_outputPrewitt, 0, pixel_mem)); //make sure no memory is left laying around

  checkCudaErrors(cudaMalloc(d_outputCross, pixel_mem));
  checkCudaErrors(cudaMemset(*d_outputCross, 0, pixel_mem)); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImage, *h_inputImage, pixel_mem, cudaMemcpyHostToDevice));

  d_inputImage__  = *d_inputImage; // DJM - not sure what these are for?
  d_outputSobel__ = *d_outputSobel;
  d_outputPrewitt__ = *d_outputPrewitt;
  d_outputCross__ = *d_outputCross;

  //now create the filter that they will use
  // DJM - maybe we should create the filters we are using here?
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 4.; // changes how blurred image becomes

  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
}
}

void postProcess(const std::string& sobel_file, const std::string& prewitt_file,
                 const std::string& cross_file)
{
  cv::imwrite(sobel_file.c_str(), imageSobel);
  cv::imwrite(prewitt_file.c_str(), imagePrewitt);
  cv::imwrite(cross_file.c_str(), imageCross);

  // can be uncommented to automatically show input/output images
  /*cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", imageInput);
  cv::waitKey(0);
  cv::imshow("Display window", imageSobel);
  cv::waitKey(0);
  cv::imshow("Display window", imagePrewitt);
  cv::waitKey(0);
  cv::imshow("Display window", imageCross);
  cv::waitKey(0);*/
}

void cleanUp(void)
{
  cudaFree(d_inputImage__);
  cudaFree(d_outputSobel__);
  cudaFree(d_outputPrewitt__);
  cudaFree(d_outputCross__);
  delete[] h_filter__;
}
