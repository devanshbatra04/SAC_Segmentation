#include <opencv2/viz.hpp>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#include <numeric>
#include <cmath>
#include <string>
#include "./sac_segmentation.cpp"
#define PLANE_MODEL 1;
#define SAC_METHOD_RANSAC 1;

using namespace cv;
using namespace std;

int main () {
    Mat cloud = cv::viz::readCloud("/home/groot/GSoC20/data/CobbleStones.obj");
    // const Vec3d p1(cloud.ptr<double>(i));

    viz::Viz3d window("window");
    window.showWidget("Coordinate Widget", viz::WCoordinateSystem());
    viz::WCloud cloud_widget1(cloud);
    // viz::WPlane plane( Size2d(10, 10), viz::Color::green());
    window.showWidget("cloud 1", cloud_widget1);
    // window.showWidget("plane", plane);

    UMat rendered;
    // pca.df.render(rendered, cloud);
    // imshow("render", cloud);    
    
    const Vec3f* points = cloud.ptr<Vec3f>(0);
    const unsigned num_points = cloud.cols;    
    // for(int j = 0; j < cloud.cols; j++)
        // cout << points[j][0] << " " << points[j][1] << " " << points[j][2] << endl;

    std::vector<unsigned> indices(num_points);
    std::iota(std::begin(indices), std::end(indices), 0);

    // get minimum inliers for model
    vector<unsigned> inliers_indices;
    const unsigned num_rnd_model_points = 3;
    RNG rng((uint64)-1);

    int num_planes = 20;
    vector<viz::WPlane> PlaneWidgets;
    while(num_planes--) {
        inliers_indices.clear();
        for (int i = 0; i < num_rnd_model_points; ++i) {
                std::swap(indices[i], indices[rng.uniform(0, num_points)]);
        }

        for (unsigned i = 0; i < num_rnd_model_points; i++) {
            unsigned idx = indices[i];
            inliers_indices.emplace_back(idx);
        }

        Point3d center;
        Vec4d coefficients = getPlaneFromPoints(points, inliers_indices, center);

        // Add coefficients of found plane to new class SAC plane and add it to window.
        
        cv::SACSegmentation::SACPlaneModel SACplane (coefficients, center);
        // SACplane.addToWindow(window);
        viz::WPlane widget = SACplane.WindowWidget(window);
        PlaneWidgets.push_back(widget);
    }
    int num_planes_shown = 0;
    for (viz::WPlane planeWidget: PlaneWidgets) {
        window.showWidget("plane" + to_string(num_planes_shown++), planeWidget);
    }
    window.spin();
    waitKey(1);
    return 0;
}