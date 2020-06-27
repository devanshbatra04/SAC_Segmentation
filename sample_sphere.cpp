#include <opencv2/viz.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <string>
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "./sac_segmentation/sac_segmentation.hpp"

using namespace cv;
using namespace std;

int main() {
    // Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/wallnut.ply");
    Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/sphere-big.obj");
    cv::SACSegmentation::SACModelFitting sphere_segmentation(cloud, SPHERE_MODEL);
    /// Adds cloud to window
    viz::Viz3d window("original cloud");
    viz::WCloud cloud_widget1(cloud);
    window.showWidget("cloud 1", cloud_widget1);

    sphere_segmentation.threshold = 0.0001;
    sphere_segmentation.max_iters=10000;
    viz::Viz3d fitted("fitted cloud");
   
    // sphere_segmentation.window=fitted;
    sphere_segmentation.fit_once();
    cout << sphere_segmentation.model_instances.at(0).ModelCoefficients.size();
    cv::SACSegmentation::SACSphereModel sphere (sphere_segmentation.model_instances.at(0));
    cout << sphere.center << endl;
    cout << sphere.radius << endl;
    cout << sphere_segmentation.inliers.at(0).size() << endl;
    sphere.radius *= 0.75;
    sphere.addToWindow(window);

    // // fitted.showWidget("model", widget);

    window.spin();
    waitKey(1);
    
}