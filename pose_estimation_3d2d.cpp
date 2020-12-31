//
// Created by q on 2020/12/30.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

#include <Eigen/Core>
//#include <g2o/core/base_vertex.h>
//#include <g2o/core/base_unary_edge.h>
//#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h>

#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>

using namespace std;
typedef vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d> VecVector3d;
void bundleAdjustmentG2o(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &k
);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);
void img_matches(
        cv::Mat &img1,cv::Mat &img2,
        vector<cv::KeyPoint> &key_point1,
        vector<cv::KeyPoint> &key_point2,
        vector<cv::DMatch> &good_matches);
void pos_estimator_2d2d(vector<cv::KeyPoint>&key_point1,
                        vector<cv::KeyPoint>&key_point2,
                        vector<cv::DMatch>&matches,
                        cv::Mat &R,cv::Mat &t);

int main()
{
    cv::Mat img1=cv::imread("../1.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat img2=cv::imread("../2.png",CV_LOAD_IMAGE_COLOR);
    cv::Mat depth_1 = cv::imread("../1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depth_2 = cv::imread("../2_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    assert(img1.data != nullptr && img2.data!= nullptr);//为真时程序正常运行

    //提取特征点
    vector<cv::KeyPoint>keypoint1,keypoint2;
    vector<cv::DMatch>matches;
    img_matches(img1,img2,keypoint1,keypoint2,matches);
    cout<<"find matches "<<matches.size()<<endl;

    //建立3D点
    cv::Mat k = (cv::Mat_<double>(3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point3d>pts_3d;
    vector<cv::Point2d>pts_2d;
    for(auto &p:matches)
    {
        ushort d = depth_1.at<unsigned short >(keypoint1[p.queryIdx].pt.y,keypoint1[p.queryIdx].pt.x);//
        ushort d1 = depth_1.ptr<unsigned short>(int(keypoint1[p.queryIdx].pt.y))[int(keypoint1[p.queryIdx].pt.x)];//两种图像寻址方式都行
        if(d==0)
            continue;
        double d5 = d / 5000.0;
        cv::Point2d p1 = pixel2cam(keypoint1[p.queryIdx].pt,k);//归一化坐标
        pts_3d.emplace_back(p1.x*d5, p1.y*d5, d5);
        pts_2d.emplace_back(keypoint2[p.trainIdx].pt);
        cout<<"depth= "<<d<<"  d1= "<<d1<<"  pts_3d x="<<pts_3d.data()->x<<"  y="<<pts_3d.data()->y<<"  z"<<pts_3d.data()->z<<endl;
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::Mat r,t;
    cv::solvePnP(pts_3d,pts_2d,k,cv::Mat(),r,t,false);//opencv PnP
    cv::Mat R;
    cv::Rodrigues(r,R);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_use = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"OpenCV pnp time used"<<time_use.count()<<" second"<<endl;
    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;

    cv::Mat d2_R,d2_t;
    pos_estimator_2d2d(keypoint1,keypoint2,matches,d2_R,d2_t);
    cout << "d2_R=" << endl << d2_R << endl;
    cout << "d2_t=" << endl << d2_t << endl;
    cout << "t/d2_t"<< endl << t/d2_t<<endl;//用2d和3d做对比

    //BA G2o
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for(auto i=0;i<pts_3d.size();i++)
    {
        pts_3d_eigen.push_back(Eigen::Vector3d (pts_3d[i].x,pts_3d[i].y,pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d (pts_2d[i].x,pts_2d[i].y));
    }

    return 0;
}
void bundleAdjustmentG2o(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const cv::Mat &k,
        Sophus::SE3d &pose
        )
{
#if 0  //举例
    g2o::SparseOptimizer optimizer;//这就是图
    //使用cholmod中的线性方程求解器 维度为 6 ，3  ；
    // linearSolver：Hx=-b求解器
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver=new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
    //6*3的参数
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3(linearSolver);//blocksolver 单个误差项对应参数块
    //L-M 下降方法
    g2o::OptimizationAlgorithmLevenberg* algorithm =new g2o::OptimizationAlgorithmLevenberg(block_solver);
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);
    //下面添加定点和边就可以了
#endif
// 构建图优化，先设定g2o
//    typedef g2o::BlockSolver< g2o::BlockSolverTraits<3,1> > Block;  // 每个误差项优化变量维度为3，误差值维度为1
//    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>(); // 线性方程求解器
//    Block* solver_ptr = new Block( linearSolver );      // 矩阵块求解器

    //构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolveType;//SE3 pose is 6,landmark is 3
//    typedef g2o::LinearSolverDense<BlockSolveType::PoseMatrixType>linearSolverType;//线性求解器类型
    BlockSolveType::LinearSolverType* linearSolver (new g2o::LinearSolverDense<BlockSolveType::PoseMatrixType>()); // 线性方程求解器
    BlockSolveType* solver_ptr =new BlockSolveType( std::move(linearSolver));      // 矩阵块求解器
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
}

//计算归一化坐标
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void img_matches(cv::Mat &img1,cv::Mat &img2,vector<cv::KeyPoint> &key_point1,vector<cv::KeyPoint> &key_point2,vector<cv::DMatch> &good_matches) {
    assert(img1.data != nullptr || img2.data != nullptr);

    //数据初始化
//    vector<cv::KeyPoint> key_point1,key_point2;//特征点向量
    cv::Mat descriptor_1, descriptor_2;//描述子数组
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();//特征检测器
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();//描述符提取器
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");//描述符匹配器

    //检测 oriented FAST角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img1, key_point1);
    detector->detect(img2, key_point2);

    //计算描述子
    descriptor->compute(img1, key_point1, descriptor_1);
    descriptor->compute(img2, key_point2, descriptor_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
    cout << "time ORB cost " << time_used.count() << "seconds" << endl;

    cv::Mat outimg;
    cv::drawKeypoints(img1, key_point1, outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("orb features", outimg);
    cv::waitKey(1);

    //匹配
    vector<cv::DMatch> matchs;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptor_1, descriptor_2, matchs);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
    cout << "time match cost " << time_used.count() << "seconds" << endl;
    //匹配点筛选
    auto min_max = minmax_element(matchs.begin(), matchs.end(),
                                  [](const cv::DMatch &m1, const cv::DMatch &m2) {
                                      return m1.distance < m2.distance;
                                  });//使用lambda函数自定义比较
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

//    vector<cv::DMatch>good_matches;
    for (auto &p:matchs) {
        if (p.distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(p);
        }
    }
    //绘制结果
    cv::Mat img_matches, img_good_matches, img_goodkey1, img_goodkey2;
    vector<cv::KeyPoint> key_point1_d, key_point2_d;
    for (auto &p:good_matches)//match函数的参数中位置在前面的为query descriptor，后面的是 train descriptor
    {
        key_point1_d.push_back(key_point1[p.queryIdx]);
        key_point2_d.push_back(key_point2[p.trainIdx]);
    }
    cv::drawMatches(img1, key_point1, img2, key_point2, matchs, img_matches);
    cv::drawMatches(img1, key_point1, img2, key_point2, good_matches, img_good_matches);
    cv::drawKeypoints(img1, key_point1_d, img_goodkey1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(img2, key_point2_d, img_goodkey2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


    cv::imshow("matches", img_matches);
    cv::imshow("good matches", img_good_matches);
    cv::imshow("img_goodkey1", img_goodkey1);
    cv::imshow("img_goodkey2", img_goodkey2);
    cv::waitKey(1);
    std::cout << "Hello, World!" << std::endl;
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
//        cout<<" d="<<d<<endl;
    }
}