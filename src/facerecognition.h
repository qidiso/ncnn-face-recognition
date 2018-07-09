#pragma once
#ifndef FACERECOGNITION_H_
#define FACERECOGNITION_H_

#include <iostream>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "net.h"
#include "mobilefacenet.h"
#include "mtcnn.h"
#include "featuredb.h"
#include <string>
#include <opencv2/opencv.hpp>

#include "boost/python.hpp"

namespace bp = boost::python;
using namespace std;

struct AlignedFace
{
    int rect[4];
    cv::Mat face;
};

cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst);

class FaceRecognition
{
public:
    FaceRecognition(bp::str str, float thres);
    ~FaceRecognition();

    bp::list recognize(int rows,int cols,bp::str img_data);
    int add_person(bp::str str, int rows,int cols,bp::str img_data);
    int del_person(bp::str str);
    bp::list get_names();

private:
    std::string modulepath;
    MTCNN *mtcnn;
    MobileFaceNet *mobilefacenet;
    FeatureDB *featuredb;

    int align(cv::Mat image, std::vector<AlignedFace> &aligned_face);
};

#endif /* FACERECOGNITION_H_ */
