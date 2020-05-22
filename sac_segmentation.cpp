#include <opencv2/viz.hpp>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#define PLANE_MODEL 1;
#define SAC_METHOD_RANSAC 1;

using namespace cv;
using namespace std;

namespace cv {
namespace SACSegmentation {

class SACModel {
    public:
        vector<float> ModelCoefficients;
        // vector<Point3f> inliers;
        virtual ~SACModel() 
        { 

        }
        virtual void addToWindow(viz::Viz3d & window) {
        }
        virtual void getModelFromPoints(Mat inliers) {

        };
};

class SACPlaneModel : public SACModel {
    private:
        Point3d center;
        Vec3f normal;
        Size2d size;
        
    public:
        SACPlaneModel() {
            // Vec4f & cf = this->ModelCoefficients;
        } 
        SACPlaneModel(vector<float> coefficients, Size2d size=Size2d(10.0, 10.0)) {
            assert(coefficients.size() == 4);
            this->ModelCoefficients = coefficients;
            this->size = size;
            
            // Assign normal vector
            for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

            center.x = 0;
            center.y = 0;
            center.z = - coefficients[3] / coefficients[4];
        }
        void addToWindow(viz::Viz3d & window) {
            viz::WPlane plane( this->size, viz::Color::green());
            window.showWidget("plane", plane);
        }
};

// class PnPRansacCallback: public PointSetRegistrator::Callback {

// };

class SACModelFitting {
    public:
        int model_type = PLANE_MODEL;
        int method_type = SAC_METHOD_RANSAC;
        int threshold;
        int max_iters = 1000;
        double _confidence=0.99;
        cv::Mat remainingCloud;
        vector<Mat> inliers;
        vector<SACModel> model_instances;

        // Get one model (plane), this function would call RANSAC on the given set of points and get the biggest model (plane).
        void fit_once() {
            // Ptr<PointSetRegistrator::Callback> cb; // pointer to callback
            // if (model_type != PLANE_MODEL || method_type != SAC_METHOD_RANSAC) {
            //     cout << "Model or method not yet supported";
            //     return;
            // }
            // cb = makePtr<PnPRansacCallback>();
        }
        
        // 
};

}
}