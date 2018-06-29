#pragma once
#ifndef MOBILEFACENET_H_
#define MOBILEFACENET_H_
#include <string>
#include "net.h"
#include "opencv2/opencv.hpp"


class MobileFaceNet {
public:
	MobileFaceNet(const std::string &model_path);
	~MobileFaceNet();
	void start(const cv::Mat& img, std::vector<float>&feature);
private:
	void RecogNet(ncnn::Mat& img_);
	ncnn::Net Recognet;
	ncnn::Mat ncnn_img;
	std::vector<float> feature_out;
};

double calculSimilar(std::vector<float>& v1, std::vector<float>& v2);


#endif // !MOBILEFACENET_H_
