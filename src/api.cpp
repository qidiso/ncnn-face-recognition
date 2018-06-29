#include <iostream>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "net.h"
#include "mobilefacenet.h"
#include "mtcnn.h"
#include <string>
#include <opencv2/opencv.hpp>

#include "boost/python.hpp"
namespace bp = boost::python;


using namespace std;

std::vector<std::string> splitString_1(const std::string &str,
    const char delimiter) {
    std::vector<std::string> splited;
    std::string s(str);
    size_t pos;

    while ((pos = s.find(delimiter)) != std::string::npos) {
        std::string sec = s.substr(0, pos);

        if (!sec.empty()) {
            splited.push_back(s.substr(0, pos));
        }

        s = s.substr(pos + 1);
    }

    splited.push_back(s);

    return splited;
}

double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}



cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
   int size = dst.size();
   cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
   cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

   for (int i = 0; i < size; i++)
   {
      A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
      A.at<float>(i << 1, 1) = -x0[i].y;
      A.at<float>(i << 1, 2) = 1;
      A.at<float>(i << 1, 3) = 0;
      A.at<float>(i << 1 | 1, 0) = x0[i].y;
      A.at<float>(i << 1 | 1, 1) = x0[i].x;
      A.at<float>(i << 1 | 1, 2) = 0;
      A.at<float>(i << 1 | 1, 3) = 1;

      B.at<float>(i << 1) = dst[i].x;
      B.at<float>(i << 1 | 1) = dst[i].y;
   }

   cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
   cv::Mat AT = A.t();
   cv::Mat ATA = A.t() * A;
   cv::Mat R = ATA.inv() * AT * B;

   roi.at<float>(0, 0) = R.at<float>(0, 0);
   roi.at<float>(0, 1) = -R.at<float>(1, 0);
   roi.at<float>(0, 2) = R.at<float>(2, 0);
   roi.at<float>(1, 0) = R.at<float>(1, 0);
   roi.at<float>(1, 1) = R.at<float>(0, 0);
   roi.at<float>(1, 2) = R.at<float>(3, 0);
   return roi;

}

class NoFaceException: public exception
{
  virtual const char* what() const throw()
  {
    return "No face found";
  }
} noface;


int test(int argc, char* argv[])
{
//    MTCNN mtcnn(model_path);

    //fstream in("pairs_1.txt");
    //fstream out("rs_lfw_99.50.txt",ios::out);

//    cv::Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    //cv::Mat img2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

//    cv::Mat alignedFace1 = faceAlign(img1, &mtcnn);
    //cv::Mat alignedFace2 = faceAlign(img2, &mtcnn);

 //   std::vector<float> feature1;
    //std::vector<float> feature2;
    //recognize.start(alignedFace1, feature1);
    //test_camera(&mtcnn, feature1);
    //test_camera(&mtcnn);
    //recognize.start(alignedFace2, feature2);
    //double sim = calculSimilar(feature1, feature2);
    //cout <<sim << endl;
#if 0
        string line;
    while (in >> line)
    {
        //cout <<line<<endl;
        std::vector<std::string>  rs = splitString_1(line, ',');

        string img_L = rs[0];
        string img_R = rs[1];
        string flag = rs[2];
        //cout <<img_L<<endl;
        std::vector<float> cls_scores;
        cv::Mat img1 = cv::imread("../lfw-112X112/" + img_L, CV_LOAD_IMAGE_COLOR);
        cv::Mat img2 = cv::imread("../lfw-112X112/" + img_R, CV_LOAD_IMAGE_COLOR);

        //cout << "lfw-112X112/" + img_L << endl;

        double start = get_current_time();
        recognize.start(img1, feature1);
        recognize.start(img2, feature2);
        double end = get_current_time();
        double total_time = (double)(end - start);
        std::cout << "time=" << total_time << "ms" << std::endl;



        double sim = calculSimilar(feature1, feature2);
        //fprintf(stderr, "%s,%f\n", flag.c_str(), sim);
        out << flag.c_str() << "\t"<<sim << endl;
    }

#endif
    //cv::imshow("left", img1);
    //cv::imshow("right", img2);
    //cv::waitKey(0);

    return 0;
}


class facerecognition{
public:
    facerecognition(bp::str str) {
        modulepath = std::string(((const char *) bp::extract<const char *>(str)));
        mtcnn = new MTCNN(modulepath);
        mobilefacenet = new MobileFaceNet(modulepath);
        std::cout << "communication init." << std::endl;
        std::cout << "modulepath is " << modulepath << std::endl;
    }

    bp::str recognize(int rows,int cols,bp::str img_data) {
        unsigned char *data = (unsigned char *) ((const char *) bp::extract<const char *>(img_data));
        cv::Mat img= cv::Mat(rows, cols, CV_8UC3,data);
        cv::Mat img2= cv::imread("1.jpg", CV_LOAD_IMAGE_COLOR);;

	cv::Mat aligned_face1 = align(img);
	cv::Mat aligned_face2 = align(img2);

        std::vector<float> feature1;
        std::vector<float> feature2;
        mobilefacenet->start(aligned_face1, feature1);
        mobilefacenet->start(aligned_face2, feature2);
        double sim = calculSimilar(feature1, feature2);
        cout << "sim " << sim << endl;
        return "gf";
    }

private:
    std::string modulepath;
    MTCNN *mtcnn;
    MobileFaceNet *mobilefacenet;

    cv::Mat align(cv::Mat image){

        double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
        vector<cv::Point2f>coord5points;
        vector<cv::Point2f>facePointsByMtcnn;
        for (int i = 0; i < 5; i++) {
                coord5points.push_back(cv::Point2f(dst_landmark[i], dst_landmark[i + 5]));
        }

        cout << image.cols << image.rows << endl;

        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
        std::vector<Bbox> bboxes;

#if(MAXFACEOPEN==1)
        mtcnn->detectMaxFace(ncnn_img, bboxes);
#else
        mtcnn->detect(ncnn_img, bboxes);
#endif

        const int num_box = bboxes.size();
        cout << num_box << endl;
        if (num_box == 0)
            throw noface;
        std::vector<cv::Rect> bbox;
        bbox.resize(num_box);
        for (int i = 0; i < num_box; i++) {
            bbox[i] = cv::Rect(bboxes[i].x1, bboxes[i].y1,
			   bboxes[i].x2 - bboxes[i].x1 + 1, bboxes[i].y2 - bboxes[i].y1 + 1);

            for (int j = 0; j<5; j = j + 1){
                facePointsByMtcnn.push_back(cvPoint(bboxes[i].ppoint[j], bboxes[i].ppoint[j + 5]));
            }
        }

        cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
        if (warp_mat.empty()) {
                warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
        }
        warp_mat.convertTo(warp_mat, CV_32FC1);
        cv::Mat alignedFace = cv::Mat::zeros(112, 112, image.type());
        warpAffine(image, alignedFace, warp_mat, alignedFace.size());

    return alignedFace;
    }


#if 0
void test_camera(MTCNN *mtcnn, std::vector<float> featurein) {
        cv::VideoCapture mVideoCapture(0);
        mVideoCapture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        mVideoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
        if (!mVideoCapture.isOpened()) {
                return;
        }
        cv::Mat frame;
        mVideoCapture >> frame;
        while (!frame.empty()) {
        usleep(100 * 1000);
                mVideoCapture >> frame;
                if (frame.empty()) {
                        break;
                }

        try{
                cv::Mat alignedFace = faceAlign(frame, mtcnn);
        std::vector<float> feature;
        recognize.start(alignedFace, feature);
        double sim = calculSimilar(featurein, feature);
        cout <<sim << endl;
        } catch (exception& e){
            continue;
        }
        }
        return ;
}
#endif

};

BOOST_PYTHON_MODULE (facerecognition) {
    bp::class_<facerecognition>("facerecognition", bp::init<bp::str>())
            .def("recognize", &facerecognition::recognize);
}
