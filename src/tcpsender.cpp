#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#define _PORT_ 9999
#define _BACKLOG_ 10

using namespace std;

#include <sys/time.h>
double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#include "mobilefacenet.h"
#include "mtcnn.h"
#include "featuredb.h"

cv::Mat getsrc_roi2(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
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

struct RecognitionResult{
	cv::Rect rect;
	std::string name;

};

std::vector<RecognitionResult>  face_recognition(cv::Mat &image, MTCNN *mtcnn, MobileFaceNet* mobilefacenet, FeatureDB* featuredb)
{
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    std::vector<Bbox> bboxes;
    std::vector<RecognitionResult> ret;

    mtcnn->detect(ncnn_img, bboxes);
    //mtcnn->detectMaxFace(ncnn_img, bboxes);

    const int num_box = bboxes.size();
    for (int i = 0; i < num_box; i++) {
        cv::Mat face;
        vector<cv::Point2f> coord5points;
        vector<cv::Point2f> facePointsByMtcnn;

        cv::Rect rect(bboxes[i].x1, bboxes[i].y1, bboxes[i].x2 - bboxes[i].x1 + 1, bboxes[i].y2 - bboxes[i].y1 + 1);

        for (int j = 0; j < 5; j ++) {
            facePointsByMtcnn.push_back(cvPoint(bboxes[i].ppoint[j], bboxes[i].ppoint[j + 5]));
            coord5points.push_back(cv::Point2f(dst_landmark[j], dst_landmark[j + 5]));
        }

        cv::Mat warp_mat = estimateRigidTransform(facePointsByMtcnn, coord5points, false);
        if (warp_mat.empty()) {
            warp_mat = getsrc_roi2(facePointsByMtcnn, coord5points);
        }
        warp_mat.convertTo(warp_mat, CV_32FC1);
        face = cv::Mat::zeros(112, 112, image.type());
        warpAffine(image, face, warp_mat, face.size());

	std::vector<float> feature;
        mobilefacenet->start(face, feature);
	std::string name = featuredb->find_name(feature);

	RecognitionResult aaa;
	aaa.rect = rect;
	aaa.name = name;
	ret.push_back(aaa);
    }

    return ret;
}

#define CAMERA_NUMBER 3
std::string url[] = {"rtsp://admin:a12345678@192.168.1.12/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.17/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.18/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.19/h264/ch1/sub/av_stream"};
cv::Mat globalframe[CAMERA_NUMBER];
bool finished = false;

void* cv_thread(void *parm)
{
    int index = *(int*)parm;
    printf("===========%d=====\n", index);

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(index + 1,&mask);
    printf("thread %u, i = %d\n", pthread_self(), index + 1);
    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))
    {
        fprintf(stderr, "pthread_setaffinity_np erro\n");
        return NULL;
    }

    std::string modulepath = "/root/livemediastreamer/build/models";
    MTCNN *mtcnn2 = new MTCNN(modulepath);
    MobileFaceNet *mobilefacenet = new MobileFaceNet(modulepath);
    FeatureDB *featuredb = new FeatureDB(modulepath, 0.63);
    printf("===========%d==222===\n", index);
    cv::VideoCapture cap(url[index]);
    finished = true;
    cout << url[index] << "Opened" << endl;


	bool need = true;
	std::vector<RecognitionResult> ret;
    while(1) {
        cv::Mat frame;
        int capret = cap.read(frame);

	if (capret == 0)
	    printf("read error");
//        else
//	    cout <<  url[index] << "...Readed" << endl;
        if (need)
   	    ret = face_recognition(frame, mtcnn2, mobilefacenet, featuredb);
	need = !need;

	for(int i = 0; i < ret.size(); i ++) {
  	cv::putText(frame, ret[i].name, ret[i].rect.tl(), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255), 2, 8, 0);
        cv::rectangle(frame, ret[i].rect, cv::Scalar(0,0,255),3,1,0);
	}
        globalframe[index] = frame.clone();
    }
}

int main()
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(3,&mask);
    printf("thread %u, i = %d\n", pthread_self(), 0);
    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))
    {
        fprintf(stderr, "pthread_setaffinity_np erro\n");
        return NULL;
    }

    int sock=socket(AF_INET,SOCK_STREAM,0);
    if (sock < 0) {
        printf("socket()\n");
    }
    struct sockaddr_in server_socket;
    struct sockaddr_in socket;
    server_socket.sin_family=AF_INET;
    server_socket.sin_addr.s_addr=htonl(INADDR_ANY);
    server_socket.sin_port=htons(_PORT_);
    if(bind(sock, (struct sockaddr*)&server_socket, sizeof(struct sockaddr_in)) < 0) {
        printf("bind()\n");
        close(sock);
        return 1;
    }
    if(listen(sock, _BACKLOG_) < 0) {
        printf("listen()\n");
        close(sock);
        return 2;
    }
    //cv::VideoCapture cap1("rtsp://admin:a12345678@192.168.1.12/h264/ch1/sub/av_stream");
    //cv::VideoCapture cap2("rtsp://admin:a12345678@192.168.1.17/h264/ch1/sub/av_stream");
    pthread_t t[CAMERA_NUMBER];

    for (int i = 0; i < CAMERA_NUMBER; i ++) {
	finished = false;
        if(pthread_create(&t[i], NULL, cv_thread, (void*)&i) == -1){
            puts("fail to create pthread t0");
            exit(1);
        }

	while (!finished);
    }
    
    printf("listen success\n");
#if 0
    while(1) {
        cv::Mat frame1;
        cv::Mat frame2;
        cap1.read(frame1);
        cap2.read(frame2);

        cv::Mat f1clone = frame1.clone();

        if(frame1.type() == CV_8UC3 && frame2.type() == CV_8UC3) {
            alignFce2(f1clone);
            alignFce2(f2clone);
            cv::Mat frame;
            frame.push_back(f1clone);
            frame.push_back(f2clone);
            cv::cvtColor(frame, frame, CV_BGR2YUV_I420);
        } else {
            usleep(100 * 1000);
            printf("Error reading \n");
        }
    }
#endif
    socklen_t len=0;
    int client_sock=accept(sock, (struct sockaddr*)&socket, &len);
    if(client_sock < 0) {
        printf("accept()\n");
        return 3;
    }
    printf("get connect\n");
    char buf[4];


#if 1
    while(1) {
//         printf("start sending 111---\n");
        int ret = read(client_sock, buf, 4);
        if (ret == 4 && buf[0] == 0x11 && buf[1] == 0x22 && buf[2] == 0x33 && buf[3] == 0x44) {
    	//	printf("start sending---\n");
        } else {
//         printf("start sending 222---\n");
	    continue;
	}
resend:
#if 0
        int ret1 = cap1.read(frame1);
        int ret2 = cap2.read(frame2);
	cout << ret1 << ret2 << endl;

	cv::Mat f1clone = frame1.clone();
	cv::Mat f2clone = frame2.clone();

        if(frame1.type() == CV_8UC3 && frame2.type() == CV_8UC3) {
        //if(frame1.type() == CV_8UC3 ) {
            //alignFce2(f1clone);
            //alignFce2(f2clone);
            cv::Mat frame;
            frame.push_back(f1clone);
            frame.push_back(f2clone);
            cv::cvtColor(frame, frame, CV_BGR2YUV_I420);
            //cv::resize(yuvImg, yuvImg, cv::Size(),0.5,0.5);

            int ret = write(client_sock, frame.data, frame.rows * frame.cols);
            printf("wait...ret =%d, %f\n", ret, get_current_time());
        } else {
	    usleep(100 * 1000);
	    printf("Error reading \n");
	    goto resend;
	}
#endif
	cv::Mat combine1,combine2,frame;
	cv::Mat temp = cv::Mat::zeros(480, 640, globalframe[0].type());
	cv::hconcat(globalframe[0],globalframe[1],combine1);
	//cv::hconcat(globalframe[2],globalframe[3],combine2);
	cv::hconcat(globalframe[2], temp,combine2);
	cv::vconcat(combine1,combine2,frame);

	//cv::Mat frame;
	//frame.push_back(globalframe[0]);
	//frame.push_back(globalframe[1]);
	cv::cvtColor(frame, frame, CV_BGR2YUV_I420);
	ret = write(client_sock, frame.data, frame.rows * frame.cols);

    }
#endif
    //close(client_sock);
    close(sock);
    return 0;
}

