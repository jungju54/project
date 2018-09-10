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
#define NUM_INPUT	32
#define NUM_HIDDEN	15
#define NUM_OUTPUT	32
#define NUM_TRAINING_DATA	32*3
#define NUM_TEST_DATA	32*3
#define MAX_EPOCH	50000
#define LEARNING_RATE	0.3
#define SIGMOID(x) (1./(1+exp(-(x))))
#define NORMALIZE(x) (x/255.0)


//double training_point[NUM_TRAINING_DATA][NUM_INPUT];
//double training_target[NUM_TRAINING_DATA][NUM_OUTPUT];
//double test_point[NUM_TEST_DATA][NUM_INPUT];

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
			net_j += weight_ji[j][i] * NORMALIZE(training_point[i]);
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
			delta_kj[k][j] += -output[k] * (1 - output[k])*(NORMALIZE(training_target[k]) - output[k])*hidden[j];

	for (k = 0; k < NUM_OUTPUT; k++)
		delta_bias_k[k] += -output[k] * (1 - output[k])*(NORMALIZE(training_target[k]) - output[k]);

	//evaluate delta_ji
	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
		{
			double delta_k = 0;
			for (k = 0; k < NUM_OUTPUT; k++)
				delta_k += -output[k] * (1 - output[k])*(NORMALIZE(training_target[k]) - output[k])*weight_kj[k][j];
			delta_ji[j][i] += delta_k * hidden[j] * (1 - hidden[j])*NORMALIZE(training_point[i]);
		}

	for (j = 0; j < NUM_HIDDEN; j++)
	{
		double delta_k = 0;
		for (k = 0; k < NUM_OUTPUT; k++)
			delta_k += -output[k] * (1 - output[k])*(NORMALIZE(training_target[k]) - output[k])*weight_kj[k][j];
		delta_bias_j[j] += delta_k * hidden[j] * (1 - hidden[j]);
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
			weight_kj[k][j] -= LEARNING_RATE * delta_kj[k][j];

	for (k = 0; k < NUM_OUTPUT; k++)
		bias_k[k] -= LEARNING_RATE * delta_bias_k[k];

	for (j = 0; j < NUM_HIDDEN; j++)
		for (i = 0; i < NUM_INPUT; i++)
			weight_ji[j][i] -= LEARNING_RATE * delta_ji[j][i];

	for (j = 0; j < NUM_HIDDEN; j++)
		bias_j[j] -= LEARNING_RATE * delta_bias_j[j];

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

double hidden[NUM_HIDDEN], output[NUM_OUTPUT];
double weight_kj[NUM_OUTPUT][NUM_HIDDEN], weight_ji[NUM_HIDDEN][NUM_INPUT];
double bias_k[NUM_OUTPUT], bias_j[NUM_HIDDEN];
double delta_kj[NUM_OUTPUT][NUM_HIDDEN], delta_ji[NUM_HIDDEN][NUM_INPUT];
double delta_bias_k[NUM_OUTPUT], delta_bias_j[NUM_HIDDEN];
double error;

int main(void) {

	FILE *source, *result,*mid;

	//신경망 변수들
	

	int i, k, p;

	//이미지 축소 후 보간법으로 확대.
	IplImage *src_image = cvLoadImage("source.bmp", CV_LOAD_IMAGE_COLOR);
	CvSize SrcSize, BeforeSize, AfterSize;
	SrcSize.height = src_image->height; SrcSize.width = src_image->width;
	BeforeSize.height = src_image->height / 2; AfterSize.height = src_image->height * 2;
	BeforeSize.width = src_image->width / 2; AfterSize.width = src_image->width * 2;
	//IplImage *near_small = cvCreateImage(BeforeSize, src_image->depth, src_image->nChannels);
	IplImage *bili_small = cvCreateImage(BeforeSize, src_image->depth, src_image->nChannels);
	//IplImage *near_changed = cvCreateImage(SrcSize, src_image->depth, src_image->nChannels);
	IplImage *bili_changed = cvCreateImage(SrcSize, src_image->depth, src_image->nChannels);
	//cvResize(src_image, near_small, CV_INTER_NN);		cvResize(near_small, near_changed, CV_INTER_NN);
	cvResize(src_image, bili_small, CV_INTER_LINEAR);		cvResize(bili_small, bili_changed, CV_INTER_LINEAR);
	//cvShowImage("nearest neigbor interpolation - 1", near_small);
	//cvShowImage("nearest neigbor interpolation - 2", near_changed);
	cvShowImage("bilinear interpolation - 1 ", bili_small);
	cvShowImage("bilinear interpolation - 2", bili_changed);
	cvShowImage("source", src_image);
	cvReleaseImage(&bili_small);


	double **training_point = (double **)malloc(sizeof(double*) * 3 * src_image->height);
	for (int i = 0; i < 3 * src_image->height; i++)
		training_point[i] = (double*)malloc(sizeof(double)*src_image->width);
	double **training_target = (double **)malloc(sizeof(double*) * 3 * src_image->height);
	for (int i = 0; i < 3 * src_image->height; i++)
		training_target[i] = (double*)malloc(sizeof(double)*src_image->width);


	//확대
	IplImage *bili_big = cvCreateImage(AfterSize, src_image->depth, src_image->nChannels);
	cvResize(src_image, bili_big, CV_INTER_LINEAR);
	cvReleaseImage(&src_image);

	//신경망 초기화
	InitWeight(weight_kj, weight_ji, bias_k, bias_j);
	srand((unsigned)time(NULL));

	//기본파일, 최종파일 포인터.
	fopen_s(&source, "source.bmp", "rb");
	fopen_s(&result, "result.bmp", "wb");





	//bmp 파일 읽어오기
	BITMAPFILEHEADER header;
	BITMAPINFOHEADER info;
	fread(&header, sizeof(BITMAPFILEHEADER), 1, source);
	if (header.bfType != 0x4D42) exit(1);
	fread(&info, sizeof(BITMAPINFOHEADER), 1, source);
	BYTE *lpImg = new BYTE[info.biSizeImage];
	BYTE *bigImg = new BYTE[info.biSizeImage * 4];
	BYTE *ret = new BYTE[info.biSizeImage * 4];
	fread(lpImg, sizeof(char), info.biSizeImage, source);
	fclose(source);

	//int rwsize = WIDTHBYTES(hInfo.biBitCount * hInfo.biWidth);
	//기본이미지를 training target으로 설정.
	for (int i = 0; i < info.biHeight; i++)
		for (int j = 0; j < info.biWidth; j++) {
			training_target[3 * i][j] = 1.0*lpImg[i * 3 * info.biWidth + j * 3];
			training_target[3 * i + 1][j] = 1.0*lpImg[i * 3 * info.biWidth + j * 3 + 1];
			training_target[3 * i + 2][j] = 1.0*lpImg[i * 3 * info.biWidth + j * 3 + 2];
		}
	


	

	//축소,확대가 진행된 이미지를 training point로 설정.
	for (int i = 0; i < info.biHeight; i++)
		for (int j = 0; j < info.biWidth; j++) {
			if (bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3] < 0)
				training_point[3 * i][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3] + 256;
			else
				training_point[3 * i][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3];
			if (bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3+1] < 0)
				training_point[3 * i + 1][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3 + 1]+256;
			else
				training_point[3 * i + 1][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3 + 1];
			if (bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3 + 2] < 0)
				training_point[3 * i + 2][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3 + 2]+256;
			else
				training_point[3 * i + 2][j] = 1.0*bili_changed->imageData[(info.biHeight - 1 - i) * 3 * info.biWidth + j * 3 + 2];
		}

	
	cvReleaseImage(&bili_changed);
	//학습 진행. 신경망.
	for (int epoch = 0; epoch <= MAX_EPOCH; epoch++)
	{
		error = 0;
		ResetDelta(delta_kj, delta_ji, delta_bias_k, delta_bias_j);


		for (p = 0; p < NUM_TRAINING_DATA; p++)
		{
			Forward(training_point[p], weight_kj, weight_ji, bias_k, bias_j, hidden, output);

			for (k = 0; k < NUM_OUTPUT; k++)
				error += (output[k] - NORMALIZE(training_target[p][k]))*(output[k] - NORMALIZE(training_target[p][k])) / 2;

			Backward(training_point[p], training_target[p], hidden, output, weight_kj, delta_kj, delta_ji, delta_bias_k, delta_bias_j);
		}

		UpdateWeights(delta_kj, delta_ji, delta_bias_k, delta_bias_j, weight_kj, weight_ji, bias_k, bias_j);

		if (epoch % 1000 == 0) printf("%d: %f\n", epoch, error);
	}
	//for (int i = 0; i < 3 * info.biHeight; i++) {
	//	free(training_point[i]); free(training_target[i]);
	//}
	//free(training_point); free(training_target);


	double **test_point = (double **)malloc(sizeof(double*) * 3 * info.biHeight);
	for (int i = 0; i < 3 * info.biHeight; i++)
		test_point[i] = (double*)malloc(sizeof(double)*info.biWidth);
	for (int i = 0; i < bili_big->height; i++)
		for (int j = 0; j < bili_big->width * 3; j++) {
			if(bili_big->imageData[bili_big->width * 3 * (bili_big->height - i - 1) + j]<0)
				bigImg[i*bili_big->width * 3 + j] = bili_big->imageData[bili_big->width * 3 * (bili_big->height - i - 1) + j]+256;
			else
				bigImg[i*bili_big->width * 3 + j] = bili_big->imageData[bili_big->width * 3 * (bili_big->height - i - 1) + j];
		}
	
	//테스트
	int index = 0;
	for (int m = 0; m < 4; m++) {
		for (int i = 0; i < info.biHeight; i++) {
			for (int n = 0; n < info.biWidth; n++) {//m*info.biWidth*info.biHeight+3*n
				test_point[3 * i][n] = 1.0*bigImg[m*info.biWidth*info.biHeight*3 + i*info.biWidth*3+3 * n];
				test_point[3 * i + 1][n] = 1.0*bigImg[m*info.biWidth*info.biHeight*3 +i*info.biWidth*3+ 3 * n + 1];
				test_point[3 * i + 2][n] = 1.0*bigImg[m*info.biWidth*info.biHeight*3 + i*info.biWidth*3+3 * n + 2];
			}
		}
		for (i = 0; i < NUM_TEST_DATA; i++)
		{
			Forward(test_point[i], weight_kj, weight_ji, bias_k, bias_j,hidden, output);
			for (k = 0; k < info.biWidth;k++){
				ret[info.biWidth*info.biHeight * m * 3 + 3 * (i/3)*info.biWidth + 3*k+i%3] = (int)(255.0*output[k]);
			}
		}
	}
	//for (int i = 0; i < 3*info.biHeight; i++) {
	//	free(test_point[i]);
	//}
	//free(test_point);
	

	cvShowImage("확대", bili_big);
	
	
	cvReleaseImage(&bili_big);
	//최종이미지 생성.
	header.bfSize += info.biSizeImage * 3;
	info.biHeight = 2 * info.biHeight; info.biWidth = 2 * info.biWidth;
	info.biSizeImage = info.biSizeImage * 4;
	fwrite(&header, sizeof(char), sizeof(BITMAPFILEHEADER), result);
	fwrite(&info, sizeof(char), sizeof(BITMAPINFOHEADER), result);
	fwrite(ret, sizeof(char), info.biSizeImage, result);


	//종료.
	fclose(result);
	delete[] lpImg;
	//cvReleaseImage(&near_small);cvReleaseImage(&near_changed);
	
	
	
	delete[] bigImg;

	cvWaitKey(0);
	return 0;
}

