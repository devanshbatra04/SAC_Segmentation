#include <opencv2/viz.hpp>
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz/widgets.hpp>
#include <cassert>
#define PLANE_MODEL 1
#define SAC_METHOD_RANSAC 1

using namespace cv;
using namespace std;

namespace cv {
namespace SACSegmentation {

Vec4d getPlaneFromPoints(const Vec3f* &points,
                                   const std::vector<unsigned> &inliers, Point3d& center) {
    // REF: https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    Vec3f centroid(0, 0, 0);
    for (unsigned idx : inliers) {
        centroid += points[idx];
    }
    centroid /= double(inliers.size());

    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;

    for (size_t idx : inliers) {
        Vec3f r = points[idx] - centroid;
        xx += r(0) * r(0);
        xy += r(0) * r(1);
        xz += r(0) * r(2);
        yy += r(1) * r(1);
        yz += r(1) * r(2);
        zz += r(2) * r(2);
    }

    double det_x = yy * zz - yz * yz;
    double det_y = xx * zz - xz * xz;
    double det_z = xx * yy - xy * xy;

    Vec3d abc;
    if (det_x > det_y && det_x > det_z) {
        abc = Vec3d(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    } else if (det_y > det_z) {
        abc = Vec3d(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    } else {
        abc = Vec3d(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    }


    double magnitude_abc = sqrt(abc[0]*abc[0] + abc[1]* abc[1] + abc[2] * abc[2]);

    // Return invalid plane if the points don't span a plane.
    if (magnitude_abc == 0) {
        return Vec4d (0, 0, 0, 0);
    }
    abc /= magnitude_abc;
    double d = -abc.dot(centroid);

    Vec4d coefficients (abc[0], abc[1], abc[2], d);
    center = Point3d (centroid);
    return coefficients;
}


class SACModel {
    public:
        vector<double> ModelCoefficients;
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
        Vec3d normal;
        Size2d size = Size2d(2.0, 2.0);
    public:
        SACPlaneModel() {

        }

        SACPlaneModel(SACModel model) {
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

        SACPlaneModel(Vec4d coefficients, Point3d center, Size2d size=Size2d(2.0, 2.0)) {
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
        
        SACPlaneModel(Vec4d coefficients, Size2d size=Size2d(2.0, 2.0)) {
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

        SACPlaneModel(vector<double> coefficients, Size2d size=Size2d(2.0, 2.0)) {
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

class SACModelFitting {
    public:
        int model_type;
        int method_type;
        double threshold;
        int max_iters;
        // double _confidence=0.99;
        cv::Mat remainingCloud;
        vector<vector<unsigned>> inliers;
        vector<SACModel> model_instances;
        Mat cloud;

        SACModelFitting (Mat cloud, int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000)
            :cloud(cloud), model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}

        SACModelFitting (int model_type = PLANE_MODEL, int method_type = SAC_METHOD_RANSAC, double threshold = 20,int max_iters = 1000)
            :model_type(model_type), method_type(method_type), threshold(threshold), max_iters(max_iters) {}
        
        // Get one model (plane), this function would call RANSAC on the given set of points and get the biggest model (plane).       
        void fit_once() {

            // creates an array of indices for the points in the point cloud which will be appended as masks to denote inliers and outliers.
            const Vec3f* points = cloud.ptr<Vec3f>(0);
            unsigned num_points = cloud.cols;
            std::vector<unsigned> indices(num_points);
            std::iota(std::begin(indices), std::end(indices), 0);


            vector<unsigned> inliers_indices;
            const unsigned num_rnd_model_points = 3;
            
            // Initialize the best plane model.
            SACModel bestModel;
            pair<double, double> bestResult(0, 0);
                        
            if (model_type == PLANE_MODEL) {
                RNG rng((uint64)-1);
                for (unsigned i = 0; i < max_iters; ++i) {
                    vector<unsigned> current_model_inliers;
                    SACModel model;

                    for (int i = 0; i < num_rnd_model_points; ++i) {
                        std::swap(indices[i], indices[rng.uniform(0, num_points)]);
                    }

                    for (unsigned i = 0; i < num_rnd_model_points; i++) {
                        unsigned idx = indices[i];
                        current_model_inliers.emplace_back(idx);
                    }

                    Point3d center;
                    Vec4d coefficients = getPlaneFromPoints(points, current_model_inliers, center);  
                    cv::SACSegmentation::SACPlaneModel planeModel (coefficients, center);
                    pair<double, double> result = planeModel.getInliers(cloud, indices, threshold, current_model_inliers);

                    // Compare fitness first.
                    if (bestResult.first < result.first || (bestResult.first == result.first && bestResult.second > result.second )) {
                        bestResult = result;
                        bestModel.ModelCoefficients = planeModel.ModelCoefficients;
                        inliers_indices = current_model_inliers;
                    }
                    
                }
                inliers.push_back(inliers_indices);
                model_instances.push_back(bestModel);
            }    
        }
        
        
        // 
};

}
}