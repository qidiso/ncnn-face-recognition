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

void draw_name(cv::Mat image, cv::Rect cvrect, std::string name)
{
    cv::rectangle(image, cvrect, cv::Scalar(127,255,0),1);

    int rect[] = {cvrect.x, cvrect.y, cvrect.width, cvrect.height};
    // draw thicking corners
    int int_x = rect[2]/5;
    int int_y = rect[3]/5;
    cv::line(image, cv::Point(rect[0],rect[1]),                  cv::Point(rect[0] + int_x,rect[1]),           cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0],rect[1]),                  cv::Point(rect[0],rect[1]+int_y),             cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0],rect[1]+int_y*4),          cv::Point(rect[0],rect[1]+rect[3]),           cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0],rect[1]+rect[3]),          cv::Point(rect[0] + int_x,rect[1]+rect[3]),   cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0]+ int_x*4,rect[1]+rect[3]), cv::Point(rect[0] + rect[2],rect[1]+rect[3]), cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0] + rect[2],rect[1]+rect[3]),cv::Point(rect[0] + rect[2],rect[1]+int_y*4), cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0] + rect[2],rect[1]+int_y),  cv::Point(rect[0] + rect[2],rect[1]),         cv::Scalar(127,255,0),3);
    cv::line(image, cv::Point(rect[0] + int_x*4,rect[1]),        cv::Point(rect[0] + rect[2],rect[1]),         cv::Scalar(127,255,0),3);
    //draw middle line
    int line_x = rect[2]/8;
    cv::line(image, cv::Point(rect[0]-line_x,rect[1]+rect[3]/2),        cv::Point(rect[0] + line_x,rect[1]+rect[3]/2),      cv::Scalar(127,255,0),1);
    cv::line(image, cv::Point(rect[0]+rect[2]/2,rect[1]+rect[3]-line_x),cv::Point(rect[0]+rect[2]/2,rect[1]+rect[3]+line_x),cv::Scalar(127,255,0),1);
    cv::line(image, cv::Point(rect[0]+line_x*7,rect[1]+rect[3]/2),      cv::Point(rect[0]+line_x*9,rect[1]+rect[3]/2),      cv::Scalar(127,255,0),1);
    cv::line(image, cv::Point(rect[0]+rect[2]/2,rect[1]-line_x),        cv::Point(rect[0]+rect[2]/2,rect[1]+line_x),        cv::Scalar(127,255,0),1);
    //write name text
    cv::putText(image, name,cv::Point(rect[0]+rect[2],rect[1]), cv::FONT_HERSHEY_COMPLEX,int_y*1.0/40, cv::Scalar(242,243,231),2);
}

#define CAMERA_NUMBER 2
MTCNN *mtcnn[CAMERA_NUMBER];
MobileFaceNet *mobilefacenet[CAMERA_NUMBER];
FeatureDB *featuredb;
cv::Mat globalframe[CAMERA_NUMBER];
std::vector<RecognitionResult> result[CAMERA_NUMBER];

#define TRAINING_CMD_NULL 0
#define TRAINING_CMD_ADD 1
#define TRAINING_CMD_DEL 2

int trainingcmd = TRAINING_CMD_NULL;
char newname[1024];

//std::vector<RecognitionResult>  face_recognition(cv::Mat &image, int index)
void * face_recognition(void* parm)
{
	int index = *(int*)parm;
    double dst_landmark[10] = {
                38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
                51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
    while(1) {
   usleep(200 * 1000);
		    
    cv::Mat image = globalframe[index];
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
    std::vector<Bbox> bboxes;
    std::vector<RecognitionResult> ret;

    mtcnn[index]->detect(ncnn_img, bboxes);
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
        mobilefacenet[index]->start(face, feature);
	if (trainingcmd == TRAINING_CMD_ADD && index == 0) {
            if (featuredb->add_feature(newname, feature) == 0) {
	        trainingcmd = TRAINING_CMD_NULL;
	    }
	} else if (trainingcmd == TRAINING_CMD_DEL && index == 0) {
            featuredb->del_feature(newname);
	    trainingcmd = TRAINING_CMD_NULL;
	}
	std::string name = featuredb->find_name(feature);

	RecognitionResult aaa;
	aaa.rect = rect;
	aaa.name = name;
	ret.push_back(aaa);
    }

    result[index] = ret;
    }
}

std::string url[] = {"rtsp://admin:a12345678@192.168.1.12/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.17/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.18/h264/ch1/sub/av_stream",
		     "rtsp://admin:a12345678@192.168.1.19/h264/ch1/sub/av_stream"};
bool finished = false;

void* cv_thread(void *parm)
{
    int index = *(int*)parm;
    cv::VideoCapture cap(url[index]);
    finished = true;
    cout << "Camera " << url[index] << " Opened" << endl;

    while(1) {
        cv::Mat frame;
        int capret = cap.read(frame);

	if (capret == 0)
	    printf("read error\n");

	for(int i = 0; i < result[index].size(); i ++) {
          draw_name(frame, result[index][i].rect, result[index][i].name);
	}
        globalframe[index] = frame.clone();
    }
}

void* udp_listen_training_cmd(void*)
{
    int sockfd=socket(AF_INET,SOCK_DGRAM,0);

    struct sockaddr_in addr;
    addr.sin_family =AF_INET;
    addr.sin_port =htons(5050);
    addr.sin_addr.s_addr=inet_addr("127.0.0.1");

    int ret =bind(sockfd,(struct sockaddr*)&addr,sizeof(addr));
    if(0>ret)
    {
        printf("bind\n");
        return NULL;

    }
    struct sockaddr_in cli;
    socklen_t len=sizeof(cli);
    char buf[1024];

    printf("udp_listen_training_cmd OK\n");
    while(1)
    {
        recvfrom(sockfd,&buf,sizeof(buf),0,(struct sockaddr*)&cli,&len);

	if (strncmp(buf, "ADD:", 4) == 0) {
            trainingcmd = TRAINING_CMD_ADD;
	    strcpy(newname, buf + 4);
	} else if (strncmp(buf, "DEL:", 4) == 0) {
            trainingcmd = TRAINING_CMD_DEL;
	    strcpy(newname, buf + 4);
	} else {
            printf("Invilid Training Command %s\n",buf);
	}



    }
    close(sockfd);
        return NULL;

}

int main(int argc, char* argv[])
{
    if (argc < 3) {
	    printf("Need two urls for IP cameras\n");
    }
    url[0] = argv[1];
    url[1] = argv[2];
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
    pthread_t t[CAMERA_NUMBER];

    std::string modulepath = "./models";
    featuredb = new FeatureDB(modulepath, 0.70);
    for (int i = 0; i < CAMERA_NUMBER; i ++) {
        mtcnn[i] = new MTCNN(modulepath);
        mobilefacenet[i] = new MobileFaceNet(modulepath);
	finished = false;
        if(pthread_create(&t[i], NULL, cv_thread, (void*)&i) == -1){
            puts("fail to create pthread t0");
            exit(1);
        }
        if(pthread_create(&t[i], NULL, face_recognition, (void*)&i) == -1){
            puts("fail to create pthread t0");
            exit(1);
	}

	while (!finished);
    }
    pthread_t udpt;
    if(pthread_create(&udpt, NULL, udp_listen_training_cmd, NULL) == -1){
        puts("fail to create pthread t0");
        exit(1);
    }
    
    printf("listen success\n");
    socklen_t len=0;
    int client_sock=accept(sock, (struct sockaddr*)&socket, &len);
    if(client_sock < 0) {
        printf("accept()\n");
        return 3;
    }
    printf("get connect\n");
    char buf[4];

    cv::Mat combine1,combine2,frame;
#if 0
    cv::hconcat(globalframe[0],globalframe[1],frame);
    cv::Mat buff[8] = {frame, frame, frame, frame, frame, frame, frame,frame};
    int bufindex = 0;
#endif
    while(1) {
        int ret = read(client_sock, buf, 4);
        if (ret == 4 && buf[0] == 0x11 && buf[1] == 0x22 && buf[2] == 0x33 && buf[3] == 0x44) {
        } else {
	    continue;
	}
resend:
#if 0
	cv::Mat temp = cv::Mat::zeros(480, 640, globalframe[0].type());
	cv::hconcat(globalframe[0],globalframe[1],combine1);
	cv::hconcat(globalframe[2],globalframe[3],combine2);
	//cv::hconcat(globalframe[2], temp,combine2);
	cv::vconcat(combine1,combine2,frame);
#else
	cv::hconcat(globalframe[0],globalframe[1],frame);
//	buff[bufindex] = frame;
//	bufindex = bufindex != 7 ? bufindex + 1 : 0;
//	frame = buff[bufindex];
#endif

	cv::cvtColor(frame, frame, CV_BGR2YUV_I420);
	ret = write(client_sock, frame.data, frame.rows * frame.cols);

    }
    close(client_sock);
    close(sock);
    return 0;
}

