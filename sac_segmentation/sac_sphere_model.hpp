#include "./sac_segmentation.hpp"

using namespace cv;
using namespace std;

namespace cv {
namespace SACSegmentation {

class SACSphereModel : public SACModel {
    private:
        Point3d center;
        double radius;
    public:
        SACSphereModel() {

        }

        SACSphereModel(SACModel model) {
            this->ModelCoefficients = model.ModelCoefficients;
            // this->size = Size2d(1.0, 1.0);
            cout << model.ModelCoefficients.size();
            // Assign normal vector
            for (unsigned i = 0; i < 3; i++) normal[i] = model.ModelCoefficients[i];
            if (model.ModelCoefficients[2] != 0) {
                center.x = 0;
                center.y = 0;
                center.z = -model.ModelCoefficients[3] / model.ModelCoefficients[2];
            } else if (model.ModelCoefficients[1] != 0) {
                center.x = 0;
                center.y = -model.ModelCoefficients[3] / model.ModelCoefficients[1];
                center.z = 0;
            } else if (model.ModelCoefficients[0] != 0) {
                center.x = -model.ModelCoefficients[3] / model.ModelCoefficients[0];
                center.y = 0;
                center.z = 0;
            }
        }

        SACSphereModel(Vec4d coefficients, Point3d center, Size2d size=Size2d(2.0, 2.0)) {
            this->ModelCoefficients.reserve(4);
            for (int i = 0; i < 4; i++) {
                this->ModelCoefficients.push_back(coefficients[i]);
            }
            this->size = size;
            
            this-> normal = Vec3d(coefficients[0], coefficients[1], coefficients[2]);
            this -> center = Point3d(0, 0, - coefficients[3] / coefficients[2]);
            // Assign normal vector
            for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

            this->center = center;
        }
        
        SACSphereModel(Vec4d coefficients, Size2d size=Size2d(2.0, 2.0)) {
            this->ModelCoefficients.reserve(4);
            for (int i = 0; i < 4; i++) {
                this->ModelCoefficients.push_back(coefficients[i]);
            }
            this->size = size;
            
            this-> normal = Vec3d(coefficients[0], coefficients[1], coefficients[2]);
            this -> center = Point3d(0, 0, - coefficients[3] / coefficients[2]);
            // Assign normal vector
            for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

            if (coefficients[2] != 0) {
                center.x = 0;
                center.y = 0;
                center.z = -coefficients[3] / coefficients[2];
            } else if (coefficients[1] != 0) {
                center.x = 0;
                center.y = -coefficients[3] / coefficients[1];
                center.z = 0;
            } else if (coefficients[0] != 0) {
                center.x = -coefficients[3] / coefficients[0];
                center.y = 0;
                center.z = 0;
            }
        }

        SACSphereModel(vector<double> coefficients, Size2d size=Size2d(2.0, 2.0)) {
            assert(coefficients.size() == 4);
            this->ModelCoefficients = coefficients;
            this->size = size;
            
            // Assign normal vector
            for (unsigned i = 0; i < 3; i++) normal[i] = coefficients[i];

            center.x = 0;
            center.y = 0;
            center.z = 0;
        }
        void addToWindow(viz::Viz3d & window) {
            viz::WPlane plane(this->center, this->normal, Vec3d(1, 0, 0), this->size, viz::Color::green());
            window.showWidget("plane", plane);
        }
        viz::WPlane WindowWidget () {
            cout << this->size;
            return viz::WPlane (this->center, this->normal, Vec3d(1, 0, 0), this->size, viz::Color::green());
        }

        pair<double, double> getInliers(Mat cloud, vector<unsigned> indices, const double threshold, vector<unsigned>& inliers) {
            pair<double, double> result;
            inliers.clear();
            const Vec3f* points = cloud.ptr<Vec3f>(0);
            const unsigned num_points = indices.size();
            
            double magnitude_abc = sqrt(ModelCoefficients[0]*ModelCoefficients[0] + ModelCoefficients[1]* ModelCoefficients[1] + ModelCoefficients[2] * ModelCoefficients[2]);
            
            if (magnitude_abc == 0) {
                //something is wrong
            }


            Vec4d NormalisedCoefficients (ModelCoefficients[0]/magnitude_abc, ModelCoefficients[1]/magnitude_abc, ModelCoefficients[2]/magnitude_abc, ModelCoefficients[3]/magnitude_abc);
            double fitness = 0;
            double rmse = 0;
            for (int i = 0; i < num_points; i++) {
                unsigned ind = indices[i];
                Vec4d point4d (points[ind][0], points[ind][1], points[ind][2], 1);
                double distanceFromPlane = point4d.dot(NormalisedCoefficients);
                if (abs(distanceFromPlane) > threshold) continue;
                inliers.emplace_back(ind);

                fitness+=1;
                rmse += distanceFromPlane;
            }

            unsigned num_inliers = fitness;
            if (num_inliers == 0) {
                result.first = 0;
                result.second = 0;
            } else {
                rmse /= num_inliers; 
                fitness /= num_points;

                result.first = fitness;
                result.second = rmse;
            }

            return result;
        }
};

}
}