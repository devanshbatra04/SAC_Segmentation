#include <opencv2/viz.hpp>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#include "./sac_segmentation.cpp"
#define PLANE_MODEL 1;
#define SAC_METHOD_RANSAC 1;

using namespace cv;
using namespace std;

int main () {
    Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/CobbleStones.obj");
    // const Vec3f p1(cloud.ptr<float>(i));

    cout << cloud.rows << " " << cloud.cols;
    // Ptr<cv::PointSetRegistrator::Callback> cb;

    viz::Viz3d window("window");
    // window.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    viz::WCloud cloud_widget1(cloud);
    // viz::WPlane plane( Size2d(10, 10), viz::Color::green());
    window.showWidget("cloud 1", cloud_widget1);
    // window.showWidget("plane", plane);

    UMat rendered;
    // pca.df.render(rendered, cloud);
    // imshow("render", cloud);
    
    vector<float> coefficients;
    for (unsigned i = 0; i < 4; i++) coefficients.push_back(-1);
    cv::SACSegmentation::SACPlaneModel SACplane(coefficients);
    SACplane.addToWindow(window);
    window.spin();
    waitKey(1);
    return 0;
}