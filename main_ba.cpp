#include <iostream>
//#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>

using namespace std;
int main() {
    cv::Mat img1=cv::imread("../1.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat img2=cv::imread("../2.png",CV_LOAD_IMAGE_COLOR);
    assert(img1.data != nullptr && img2.data != nullptr);

    //数据初始化
    vector<cv::KeyPoint> key_point1,key_point2;//特征点向量
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
    cv::waitKey(0);

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

    vector<cv::DMatch>good_matches;
    for(auto &p:matchs)
    {
        if (p.distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(p);
        }
    }
    //绘制结果
    cv::Mat img_matches,img_good_matches;
    cv::drawMatches(img1,key_point1,img2,key_point2,matchs,img_matches);
    cv::drawMatches(img1,key_point1,img2,key_point2,good_matches,img_good_matches);
    cv::imshow("matches",img_matches);
    cv::imshow("good matches",img_good_matches);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
