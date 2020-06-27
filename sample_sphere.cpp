#include <opencv2/viz.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <string>
#include "./sac_segmentation/sac_segmentation.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/wallnut.ply");
    cv::SACSegmentation::SACModelFitting sphere_segmentation(cloud, SPHERE_MODEL);
    /// Adds cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud cloud_widget1(cloud);
    window.showWidget("cloud 1", cloud_widget1);

    sphere_segmentation.threshold = 0.0001;
    sphere_segmentation.max_iters=10000;
    viz::Viz3d fitted("fitted cloud");
   
    // sphere_segmentation.window=fitted;
    cv::SACSegmentation::SACSphereModel sphere = sphere_segmentation.fit_once();
    sphere.radius *= 1;
    sphere.addToWindow(window);
    
    // // Adds segmented plane to window

    // const Vec3f* points = cloud.ptr<Vec3f>(0);
    // vector<unsigned> inlier_vec =  planar_segmentation.inliers.at(0);
    // cv::Mat fit_cloud(1, inlier_vec.size(), CV_32FC3);
    // for(int j=0; j<fit_cloud.cols; ++j)
    //     fit_cloud.at<Vec3f>(0, j) = points[inlier_vec.at(j)];

    
    // viz::WCloud cloud_widget2(fit_cloud);
    // fitted.showWidget("cloud 1", cloud_widget2);
    
    // cv::SACSegmentation::SACPlaneModel SACplane (planar_segmentation.model_instances.at(0));
    // sphere.addToWindow(fitted);
    // // viz::WPlane widget = SACplane.WindowWidget();
    // // fitted.showWidget("model", widget);

    window.spin();
    waitKey(1);
    
}