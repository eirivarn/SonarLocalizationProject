#include "shape_fitting.hpp"

// Function that fits a plane to a vector of 3D points
Eigen::Vector3d fit_plane(const std::vector<Eigen::Vector3d>& points) {
    int n = points.size();
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd Z(n);

    for (int i = 0; i < n; ++i) {
        A(i, 0) = points[i].x();
        A(i, 1) = points[i].y();
        A(i, 2) = 1.0;
        Z(i) = points[i].z();
    }

    Eigen::Vector3d Coefficients = A.colPivHouseholderQr().solve(Z);
    return Coefficients;
}

// Function that fits a paraboloid to a vector of 3D points
Eigen::VectorXd fit_paraboloid(const std::vector<Eigen::Vector3d>& points) {
    int n = points.size();
    Eigen::MatrixXd A(n, 6);
    Eigen::VectorXd Z(n);

    for (int i = 0; i < n; ++i) {
        double x = points[i].x();
        double y = points[i].y();
        A(i, 0) = x * x;
        A(i, 1) = y * y;
        A(i, 2) = x * y;
        A(i, 3) = x;
        A(i, 4) = y;
        A(i, 5) = 1.0;
        Z(i) = points[i].z();
    }

    Eigen::VectorXd Coefficients = A.colPivHouseholderQr().solve(Z);
    return Coefficients;
}
