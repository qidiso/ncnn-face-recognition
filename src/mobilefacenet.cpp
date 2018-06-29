/*
created by L. 2018.05.16
*/

#include "mobilefacenet.h"


MobileFaceNet::MobileFaceNet(const std::string &model_path) {
    std::string param_files = model_path + "/mobilefacenet.param";
    std::string bin_files = model_path + "/mobilefacenet.bin";
    Recognet.load_param(param_files.c_str());
    Recognet.load_model(bin_files.c_str());
}

MobileFaceNet::~MobileFaceNet() {
    Recognet.clear();
}

void MobileFaceNet::RecogNet(ncnn::Mat& img_) {
    ncnn::Extractor ex = Recognet.create_extractor();
    ex.set_num_threads(4);
    ex.set_light_mode(true);
    ex.input("data", img_);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature_out.resize(128);
    for (int j = 0; j < 128; j++) {
        feature_out[j] = out[j];
    }
}

void MobileFaceNet::start(const cv::Mat& img, std::vector<float>&feature) {
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 112, 112);
    RecogNet(ncnn_img);
    feature = feature_out;
}

