#include <opencv2/viz.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <string>
#include "./sac_segmentation.cpp"

using namespace cv;
using namespace std;

int main() {
    Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/CobbleStones.obj");
    cv::SACSegmentation::SACModelFitting planar_segmentation(cloud);
    // cout << cloud.type();
    /// Adds cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud cloud_widget1(cloud);
    window.showWidget("cloud 1", cloud_widget1);


    planar_segmentation.threshold = 0.5;
    planar_segmentation.max_iters=1;
    planar_segmentation.fit_once();

    // / Adds segmented plane to window

    const Vec3f* points = cloud.ptr<Vec3f>(0);
    vector<unsigned> inlier_vec =  planar_segmentation.inliers.at(0);
    cv::Mat fit_cloud(1, inlier_vec.size(), CV_32FC3);
    for(int j=0; j<fit_cloud.cols; ++j)
        fit_cloud.at<Vec3f>(0, j) = points[inlier_vec.at(j)];

    viz::Viz3d fitted("fitted cloud");
    viz::WCloud cloud_widget2(fit_cloud);
    fitted.showWidget("cloud 1", cloud_widget2);
   
    window.spin();
    fitted.spin();
    waitKey(1);
    
}