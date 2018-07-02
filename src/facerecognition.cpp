#include "facerecognition.h"

cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
    int size = dst.size();
    cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
    cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

    for (int i = 0; i < size; i++) {
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


FaceRecognition::FaceRecognition(bp::str str)
{
    modulepath = std::string(((const char *) bp::extract<const char *>(str)));
    mtcnn = new MTCNN(modulepath);
    mobilefacenet = new MobileFaceNet(modulepath);
    featuredb = new FeatureDB(modulepath, 0.5);
    std::cout << "FaceRecognition Init Finished." << std::endl;
}

FaceRecognition::~FaceRecognition()
{
    delete mtcnn;
    delete mobilefacenet;
}

bp::list FaceRecognition::recognize(int rows,int cols,bp::str img_data)
{
    unsigned char *data = (unsigned char *) ((const char *) bp::extract<const char *>(img_data));
    cv::Mat img= cv::Mat(rows, cols, CV_8UC3,data);
    std::vector<AlignedFace> aligned_face;
    bp::list recog_result;

    const int num = align(img, aligned_face);
    for (int i = 0; i < num; i ++) {
        std::vector<float> feature;
	std::string name;
	bp::list rect;
	bp::dict res;

	mobilefacenet->start(aligned_face[i].face, feature);
	name = featuredb->find_name(feature);
        rect.append(aligned_face[i].rect[0]);
        rect.append(aligned_face[i].rect[1]);
	rect.append(aligned_face[i].rect[2]);
	rect.append(aligned_face[i].rect[3]);
	res["name"] = name;
	res["rect"] = rect;
        recog_result.append(res);
    }

    //mobilefacenet->start(aligned_face, feature);
    //std::string name = featuredb->find_name(feature);
    //return name.c_str();
    return recog_result;
}

int FaceRecognition::add_person(bp::str str, int rows,int cols,bp::str img_data)
{
    unsigned char *data = (unsigned char *) ((const char *) bp::extract<const char *>(img_data));
    std::string name = std::string(((const char *) bp::extract<const char *>(str)));
    cv::Mat img= cv::Mat(rows, cols, CV_8UC3,data);

//    cv::Mat aligned_face = align(img);
    std::vector<float> feature;
//    mobilefacenet->start(aligned_face, feature);
//    featuredb->add_feature(name, feature);
}
    
int FaceRecognition::align(cv::Mat image, std::vector<AlignedFace> &aligned_face)
{
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    std::vector<Bbox> bboxes;

    mtcnn->detect(ncnn_img, bboxes);
    //mtcnn->detectMaxFace(ncnn_img, bboxes);

    const int num_box = bboxes.size();
    for (int i = 0; i < num_box; i++) {
	AlignedFace face;
        vector<cv::Point2f> coord5points;
        vector<cv::Point2f> facePointsByMtcnn;

        face.rect[0] = bboxes[i].x1;
        face.rect[1] = bboxes[i].y1;
	face.rect[2] = bboxes[i].x2 - bboxes[i].x1 + 1;
	face.rect[3] = bboxes[i].y2 - bboxes[i].y1 + 1;

        for (int j = 0; j < 5; j ++) {
            facePointsByMtcnn.push_back(cvPoint(bboxes[i].ppoint[j], bboxes[i].ppoint[j + 5]));
            coord5points.push_back(cv::Point2f(dst_landmark[i], dst_landmark[i + 5]));
        }

        cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
        if (warp_mat.empty()) {
            warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
        }
        warp_mat.convertTo(warp_mat, CV_32FC1);
        face.face = cv::Mat::zeros(112, 112, image.type());
        warpAffine(image, face.face, warp_mat, face.face.size());
	aligned_face.push_back(face);
    }

    return num_box;
}

BOOST_PYTHON_MODULE (facerecognition)
{
    bp::class_<FaceRecognition>("FaceRecognition", bp::init<bp::str>())
            .def("add_person", &FaceRecognition::add_person)
            .def("recognize", &FaceRecognition::recognize);

    bp::class_<RecogResult>("RecogResult");
}
