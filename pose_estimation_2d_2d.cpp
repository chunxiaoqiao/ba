//
// Created by q on 2020/12/29.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
void pos_estimator_2d2d(vector<cv::KeyPoint>&key_point1,
                        vector<cv::KeyPoint>&key_point2,
                        vector<cv::DMatch>&matches,
                        cv::Mat &R,cv::Mat &t);
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void img_matches(
        cv::Mat &img1,cv::Mat &img2,
        vector<cv::KeyPoint> &key_point_1,
        vector<cv::KeyPoint> &key_point_2,
        vector<cv::DMatch> &matches);

int main()
{
    cv::Mat img1=cv::imread("../1.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat img2=cv::imread("../2.png",CV_LOAD_IMAGE_COLOR);
    assert(img1.data != nullptr && img2.data!= nullptr);//为真时程序正常运行
    vector<cv::KeyPoint>KeyPoint_1,KeyPoint_2;
    vector<cv::DMatch> matches;
    img_matches(img1,img2,KeyPoint_1,KeyPoint_2,matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    cv::Mat R,t;
    pos_estimator_2d2d(KeyPoint_1,KeyPoint_2,matches,R,t);

    return 0;
}
void pos_estimator_2d2d(vector<cv::KeyPoint>&key_point1,
                        vector<cv::KeyPoint>&key_point2,
                        vector<cv::DMatch>&matches,
                        cv::Mat &R,cv::Mat &t)
{
    //相机内参
    cv::Mat k=(cv::Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    //把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> point1;
    vector<cv::Point2f> point2;
    for(auto &p:matches)
    {
        point1.push_back(key_point1[p.queryIdx].pt);
        point2.push_back(key_point2[p.trainIdx].pt);
    }
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(point1,point2,CV_FM_8POINT);//八点法求基础矩阵
    cout << "fundamental_matrix is \n" << endl << fundamental_matrix << endl;

    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(point1,point2,k);//求本质矩阵
    cout<<"essential_matrix is \n"<<essential_matrix<<endl;

    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(point1,point2,cv::RANSAC,3);//求单应矩阵
    cout<<"homography_matrix is \n"<<homography_matrix<<endl;

    cv::recoverPose(essential_matrix,point1,point2,k,R,t);
    cout<<"R is \n"<<R<<endl;
    cout<<"t is \n"<<t<<endl;

    cv::Mat t_;
    t_=(cv::Mat_<double>(3,3)
            <<  0,                          -t.at<double>(2, 0),    t.at<double>(1, 0),
            t.at<double>(2, 0),     0,                          -t.at<double>(0, 0),
            -t.at<double>(1, 0),   t.at<double>(0, 0),      0);
    cout<<"t^R/E is \n"<<t_*R/essential_matrix<<endl;//按位除法，opencv没有矩阵除法

    //验证对极约束
    for(auto &p:matches)
    {
        cv::Point2d pt1 = pixel2cam(key_point1[p.queryIdx].pt,k);
        cv::Point2d pt2 = pixel2cam(key_point2[p.trainIdx].pt,k);
        cv::Mat y1 = (cv::Mat_<double>(3,1)<<pt1.x,pt1.y,1);//归一化坐标
        cv::Mat y2 = (cv::Mat_<double>(3,1)<<pt2.x,pt2.y,1);
        cv::Mat d = y2.t() * t_ * R * y1;//归一化坐标x2' * t^R * x1=0
        cout<<" d="<<d<<endl;
    }
}
//计算归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}
void img_matches(cv::Mat &img1,cv::Mat &img2,vector<cv::KeyPoint> &key_point1,vector<cv::KeyPoint> &key_point2,vector<cv::DMatch> &good_matches)
{
    assert(img1.data != nullptr || img2.data != nullptr);

    //数据初始化
//    vector<cv::KeyPoint> key_point1,key_point2;//特征点向量
    cv::Mat descriptor_1,descriptor_2;//描述子数组
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();//特征检测器
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();//描述符提取器
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");//描述符匹配器

    //检测 oriented FAST角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img1,key_point1);
    detector->detect(img2,key_point2);

    //计算描述子
    descriptor->compute(img1,key_point1,descriptor_1);
    descriptor->compute(img2,key_point2,descriptor_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"time ORB cost "<<time_used.count()<<"seconds"<<endl;

    cv::Mat outimg;
    cv::drawKeypoints(img1,key_point1,outimg,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("orb features",outimg);
    cv::waitKey(1);

    //匹配
    vector<cv::DMatch> matchs;
    t1=chrono::steady_clock::now();
    matcher->match(descriptor_1,descriptor_2,matchs);
    t2=chrono::steady_clock::now();
    time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"time match cost "<<time_used.count()<<"seconds"<<endl;
    //匹配点筛选
    auto min_max = minmax_element(matchs.begin(),matchs.end(),
                                  [](const cv::DMatch &m1,const cv::DMatch &m2)
                                  {return m1.distance<m2.distance;});//使用lambda函数自定义比较
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

//    vector<cv::DMatch>good_matches;
    for(auto &p:matchs)
    {
        if (p.distance <= max(2*min_dist, 30.0))
        {
            good_matches.push_back(p);
        }
    }
    //绘制结果
    cv::Mat img_matches,img_good_matches,img_goodkey1,img_goodkey2;
    vector<cv::KeyPoint> key_point1_d,key_point2_d;
    for(auto &p:good_matches)//match函数的参数中位置在前面的为query descriptor，后面的是 train descriptor
    {
        key_point1_d.push_back(key_point1[p.queryIdx]);
        key_point2_d.push_back(key_point2[p.trainIdx]);
    }
    cv::drawMatches(img1,key_point1,img2,key_point2,matchs,img_matches);
    cv::drawMatches(img1,key_point1,img2,key_point2,good_matches,img_good_matches);
    cv::drawKeypoints(img1,key_point1_d,img_goodkey1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2,key_point2_d,img_goodkey2,cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


    cv::imshow("matches",img_matches);
    cv::imshow("good matches",img_good_matches);
    cv::imshow("img_goodkey1",img_goodkey1);
    cv::imshow("img_goodkey2",img_goodkey2);
    cv::waitKey(1);
    std::cout << "Hello, World!" << std::endl;
}