#define _CRT_SECURE_NO_WARNINGS

#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#define WIDTHBYTES(bits) (((bits)+31)/32*4)
#define BYTE    unsigned char
#define NUM_INPUT	2
#define NUM_HIDDEN	2
#define NUM_OUTPUT	1
#define NUM_TRAINING_DATA	4
#define NUM_TEST_DATA	4
#define MAX_EPOCH	100000
#define LEARNING_RATE	0.5
#define SIGMOID(x) (1./(1+exp(-(x))))


double training_point[NUM_TRAINING_DATA][NUM_INPUT]
= { { 1.0,1.0 },{ 1.0,0.0 },{ 0.0,1.0 },{ 0.0,0.0 } };

double training_target[NUM_TRAINING_DATA][NUM_OUTPUT]
= { { 0.0 },{ 1.0 },{ 1.0 },{ 0.0 } };

double test_point[NUM_TEST_DATA][NUM_INPUT]
= { { 0,0 },{ 0,1 },{ 1,0 },{ 1,1 } };

int InitWeight(double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//weight initialization
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			weight_kj[k][j] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			weight_ji[j][i] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (k = 0; k < NUM_OUTPUT; k++)
		bias_k[k] = 1.0 * (rand() % 1000 - 500) / 5000;

	for (j = 0; j < NUM_HIDDEN; j++)
		bias_j[j] = 1.0 * (rand() % 1000 - 500) / 5000;

	return 0;
}

int ResetDelta(double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//weight initialization
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			delta_kj[k][j] = 0;

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			delta_ji[j][i] = 0;

	for (k = 0; k < NUM_OUTPUT; k++)
		delta_bias_k[k] = 0;

	for (j = 0; j < NUM_HIDDEN; j++)
		delta_bias_j[j] = 0;

	return 0;
}

// generate outputs on the output nodes
int Forward(double training_point[NUM_INPUT],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN],
	double hidden[NUM_HIDDEN], double output[NUM_OUTPUT])
{
	int i, j, k;
	double net_j, net_k;

	//evaluate the output of hidden nodes
	for (j = 0; j < NUM_HIDDEN; j++)
	{
		net_j = 0;
		for (i = 0; i < NUM_INPUT; i++)
			net_j += weight_ji[j][i] * training_point[i];
		net_j += bias_j[j];
		hidden[j] = SIGMOID(net_j);
	}

	//evaluate the output of output nodes
	for (k = 0; k < NUM_OUTPUT; k++)
	{
		net_k = 0;
		for (j = 0; j < NUM_HIDDEN; j++)
			net_k += weight_kj[k][j] * hidden[j];
		net_k += bias_k[k];

		output[k] = SIGMOID(net_k);
	}

	return 0;
}

int Backward(double training_point[NUM_INPUT], double training_target[NUM_OUTPUT],
	double hidden[NUM_HIDDEN], double output[NUM_OUTPUT],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN],
	double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN])

{
	int i, j, k;

	//evaluate delta_kj
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			delta_kj[k][j] += -output[k] * (1 - output[k])*(training_target[k] - output[k])*hidden[j];

	for (k = 0; k < NUM_OUTPUT; k++)
		delta_bias_k[k] += -output[k] * (1 - output[k])*(training_target[k] - output[k]);

	//evaluate delta_ji
	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
		{
			double delta_k = 0;
			for (k = 0; k < NUM_OUTPUT; k++)
				delta_k += -output[k] * (1 - output[k])*(training_target[k] - output[k])*weight_kj[k][j];
			delta_ji[j][i] += delta_k*hidden[j] * (1 - hidden[j])*training_point[i];
		}

	for (j = 0; j < NUM_HIDDEN; j++)
	{
		double delta_k = 0;
		for (k = 0; k < NUM_OUTPUT; k++)
			delta_k += -output[k] * (1 - output[k])*(training_target[k] - output[k])*weight_kj[k][j];
		delta_bias_j[j] += delta_k*hidden[j] * (1 - hidden[j]);
	}

	return 0;
}

int UpdateWeights(double delta_kj[NUM_OUTPUT][NUM_HIDDEN], double delta_ji[NUM_HIDDEN][NUM_INPUT],
	double delta_bias_k[NUM_OUTPUT], double delta_bias_j[NUM_HIDDEN],
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//update weights
	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			weight_kj[k][j] -= LEARNING_RATE*delta_kj[k][j];

	for (k = 0; k < NUM_OUTPUT; k++)
		bias_k[k] -= LEARNING_RATE*delta_bias_k[k];

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			weight_ji[j][i] -= LEARNING_RATE*delta_ji[j][i];

	for (j = 0; j < NUM_HIDDEN; j++)
		bias_j[j] -= LEARNING_RATE*delta_bias_j[j];

	return 0;
}

int PrintWeight(double weight_kj[NUM_OUTPUT][NUM_HIDDEN], double weight_ji[NUM_HIDDEN][NUM_INPUT],
	double bias_k[NUM_OUTPUT], double bias_j[NUM_HIDDEN])
{
	int i, j, k;

	//print weights
	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			printf("%f ", weight_ji[j][i]);

	for (j = 0; j < NUM_HIDDEN; j++)
		printf("%f ", bias_j[j]);

	for (k = 0; k < NUM_OUTPUT; k++)
		for (j = 0; j < NUM_HIDDEN; j++)
			printf("%f ", weight_kj[k][j]);

	for (k = 0; k < NUM_OUTPUT; k++)
		printf("%f ", bias_k[k]);

	printf("\n");

	return 0;
}



int main(void) {

	FILE *source, *result;
	int data,prev,now,height,width;
	int line = 0, num = 0;

	double hidden[NUM_HIDDEN], output[NUM_OUTPUT];
	double weight_kj[NUM_OUTPUT][NUM_HIDDEN], weight_ji[NUM_HIDDEN][NUM_INPUT];
	double bias_k[NUM_OUTPUT], bias_j[NUM_HIDDEN];
	double delta_kj[NUM_OUTPUT][NUM_HIDDEN], delta_ji[NUM_HIDDEN][NUM_INPUT];
	double delta_bias_k[NUM_OUTPUT], delta_bias_j[NUM_HIDDEN];
	double error;

	int i, k, p;
	

	IplImage *src_image = cvLoadImage("source.bmp", CV_LOAD_IMAGE_COLOR);
	CvSize SrcSize, BeforeSize, AfterSize;
	SrcSize.height = src_image->height; SrcSize.width = src_image->width;
	BeforeSize.height = src_image->height / 2; AfterSize.height = src_image->height * 2;
	BeforeSize.width = src_image->width / 2; AfterSize.width = src_image->width * 2;
	IplImage *near_small = cvCreateImage(BeforeSize, src_image->depth, src_image->nChannels);
	IplImage *bili_small = cvCreateImage(BeforeSize, src_image->depth, src_image->nChannels);
	IplImage *near_changed = cvCreateImage(SrcSize, src_image->depth, src_image->nChannels);
	IplImage *bili_changed = cvCreateImage(SrcSize, src_image->depth, src_image->nChannels);
	cvResize(src_image, near_small, CV_INTER_NN);		cvResize(near_small, near_changed, CV_INTER_NN);
	cvResize(src_image, bili_small, CV_INTER_LINEAR);		cvResize(bili_small, bili_changed, CV_INTER_LINEAR);
	cvShowImage("nearest neigbor interpolation - 1", near_small);
	cvShowImage("nearest neigbor interpolation - 2", near_changed);
	cvShowImage("bilinear interpolation - 1 ", bili_small);
	cvShowImage("bilinear interpolation - 2", bili_changed);
	cvShowImage("source", src_image);
	
	

	InitWeight(weight_kj, weight_ji, bias_k, bias_j);
	fopen_s(&source, "source.bmp", "rb");
	fopen_s(&result, "result.bmp", "wb");
	srand((unsigned)time(NULL));
	
	
	/* �Ű��
	for (int epoch = 0; epoch <= MAX_EPOCH; epoch++)
	{
		error = 0;

		ResetDelta(delta_kj, delta_ji, delta_bias_k, delta_bias_j);

		for (p = 0; p < NUM_TRAINING_DATA; p++)
		{
			Forward(training_point[p], weight_kj, weight_ji, bias_k, bias_j,
				hidden, output);

			for (k = 0; k < NUM_OUTPUT; k++)
				error += (output[k] - training_target[p][k])*(output[k] - training_target[p][k]);

			Backward(training_point[p], training_target[p], hidden, output, weight_kj,
				delta_kj, delta_ji, delta_bias_k, delta_bias_j);
		}

		UpdateWeights(delta_kj, delta_ji, delta_bias_k, delta_bias_j,
			weight_kj, weight_ji, bias_k, bias_j);

		if (epoch % 1000 == 0) printf("%d: %f\n", epoch, error);
	}
	*/
	
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hInfo;
	//RGBQUAD hRGB[256];
	

	fread(&hf, sizeof(BITMAPFILEHEADER), 1, source);
	if (hf.bfType != 0x4D42) exit(1);
	fread(&hInfo, sizeof(BITMAPINFOHEADER), 1, source);
	//fread(hRGB, sizeof(RGBQUAD), 256, source);
	BYTE *lpImg = new BYTE[hInfo.biSizeImage];
	fread(lpImg, sizeof(char), hInfo.biSizeImage, source);
	fclose(source);
	BYTE *blue = new BYTE[hInfo.biSizeImage / 3];
	BYTE *green = new BYTE[hInfo.biSizeImage / 3];
	BYTE *red = new BYTE[hInfo.biSizeImage / 3];
	
	//int rwsize = WIDTHBYTES(hInfo.biBitCount * hInfo.biWidth);
	int dia=1, tap;
	for (int i = 0; i<hInfo.biHeight; i++) {
		for (int j = 0; j < hInfo.biWidth / 2; j++) {
			for (int k = 0; k < 3; k++) {
				lpImg[i * 3 * hInfo.biWidth + 3 * (j * 2 + dia) + k] = 255;
			}
		}
		dia = 1 - dia;
	}

	for(int i=0;i<hInfo.biHeight;i++)
		for (int j = 0; j < hInfo.biWidth; j++) {
			blue[i * hInfo.biWidth + j] = lpImg[i * 3 * hInfo.biWidth + j * 3]; 
			green[i * hInfo.biWidth + j] = lpImg[i * 3 * hInfo.biWidth + j * 3 + 1]; 
			red[i * hInfo.biWidth + j] = lpImg[i * 3 * hInfo.biWidth + j * 3 + 2];
		}
	//bilinear_interpolation(blue, hInfo.biWidth, hInfo.biHeight);
	//bilinear_interpolation(green, hInfo.biWidth, hInfo.biHeight);
	//bilinear_interpolation(red, hInfo.biWidth, hInfo.biHeight);

	for(int i=0;i<hInfo.biHeight;i++)
		for (int j = 0; j < hInfo.biWidth; j++) {
			lpImg[i * 3 * hInfo.biWidth + j * 3] = blue[i * hInfo.biWidth + j];
			lpImg[i * 3 * hInfo.biWidth + j * 3 + 1] = green[i * hInfo.biWidth + j];
			lpImg[i * 3 * hInfo.biWidth + j * 3 + 2] = red[i * hInfo.biWidth + j];
		}

	for (int i = 0; i < hInfo.biHeight; i++)
		for (int j = 0; j < hInfo.biWidth * 3; j++) {
			lpImg[i * 3 * hInfo.biWidth + j] = src_image->imageData[(hInfo.biHeight-1-i) * 3 * hInfo.biWidth + j];
		}

	fwrite(&hf, sizeof(char), sizeof(BITMAPFILEHEADER), result);
	fwrite(&hInfo, sizeof(char), sizeof(BITMAPINFOHEADER), result);
	//fwrite(hRGB, sizeof(RGBQUAD), 256, result);
	fwrite(lpImg, sizeof(char), hInfo.biSizeImage, result);
	
	fclose(result);
	cvReleaseImage(&src_image);
	cvReleaseImage(&near_small); cvReleaseImage(&bili_small);
	cvReleaseImage(&near_changed); cvReleaseImage(&bili_changed);
	delete[] lpImg;

	cvWaitKey(0);
	return 0;
}

