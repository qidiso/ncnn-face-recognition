#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>
#define _PORT_ 9999
#define _BACKLOG_ 10

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

MTCNN *mtcnn2 = NULL;
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

int alignFce2(cv::Mat &image)
{
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };

    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    std::vector<Bbox> bboxes;

    if (mtcnn2 == NULL) {
        cout << "new mtcnn" << endl;
        mtcnn2 = new MTCNN("/root/livemediastreamer/build/models");
    }
    mtcnn2->detect(ncnn_img, bboxes);
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
        cv::rectangle(image, rect, cv::Scalar(0,0,255),3,1,0);
    }

    if (num_box == 0)
            cout << "000000000000000000" << endl;

    return num_box;
}


int main()
{
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
    cv::VideoCapture cap1("rtsp://admin:a12345678@192.168.1.12/h264/ch1/sub/av_stream");
    cv::VideoCapture cap2("rtsp://admin:a12345678@192.168.1.17/h264/ch1/sub/av_stream");
    printf("listen success\n");
#if 0
    while(0) {
    
        cv::Mat frame;
        cap.read(frame);
	if(frame.type() != CV_8UC3) {
	    printf("Error reading\n");
	} else {
	    printf("reading OK\n");
	
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

    while(1) {
//         printf("start sending 111---\n");
        int ret = read(client_sock, buf, 4);
        if (ret == 4 && buf[0] == 0x11 && buf[1] == 0x22 && buf[2] == 0x33 && buf[3] == 0x44) {
//            printf("start sending---\n");
        } else {
//         printf("start sending 222---\n");
	    continue;
	}
resend:
        cv::Mat frame1;
        cv::Mat frame2;
        cap1.read(frame1);
        cap2.read(frame2);
        alignFce2(frame1);
        alignFce2(frame2);

        if(frame1.type() == CV_8UC3 && frame2.type() == CV_8UC3) {
            cv::Mat frame;
            frame.push_back(frame);
            frame.push_back(frame);
            cv::cvtColor(frame, frame, CV_BGR2YUV_I420);
            //cv::resize(yuvImg, yuvImg, cv::Size(),0.5,0.5);

            int ret = write(client_sock, frame.data, frame.rows * frame.cols);
//            printf("wait...ret =%d, %f\n", ret, get_current_time());
        } else {
	    usleep(100);
	    printf("Error reading \n");
	    goto resend;
	}

    }
    close(client_sock);
    close(sock);
    return 0;
}

