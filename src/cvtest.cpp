#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"

#include "quat.h"
#include "math_helper.h"

#include <deque>
#include <set>
#include <unordered_map>
#include <functional>
#include <fstream>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace quat;

#define PATCH_SIZE 6
#define PYRAMID_LEVELS 2

typedef Matrix<double, 6, 1> uVector;
typedef Matrix<double, 7, 1> eVector;
typedef Matrix<double, 4, 1> zVector;
typedef Matrix<double, 2, 1> pixVector;
typedef Matrix<uint8_t, PATCH_SIZE,PATCH_SIZE> patchMat;
typedef Matrix<uint8_t, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 1> multiPatchVectorI;
typedef Matrix<double, PATCH_SIZE, PATCH_SIZE*PYRAMID_LEVELS> multiPatchMatrixf;
typedef Matrix<double, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 1> multiPatchVectorf;
typedef Matrix<double, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 2> multiPatchJacMatrix;

Mat img_[PYRAMID_LEVELS];

Vector2d cam_center_;
Matrix<double, 2, 3> cam_F_;

void proj(Quat qz, Vector2d& eta, Matrix2d& jac)
{
  Vector3d zeta = qz.rota(e_z);
  Matrix3d sk_zeta = skew(zeta);
  double ezT_zeta = e_z.transpose() * zeta;
  MatrixXd T_z = T_zeta(qz);
  
  eta = cam_F_ * zeta / ezT_zeta + cam_center_;
  jac = -cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta));
}

void multiLvlPatch(const pixVector &eta, multiPatchVectorf& dst)
{
  float x = eta(0,0);
  float y = eta(1,0);
  Size sz(PATCH_SIZE, PATCH_SIZE);
  for (int i = 0; i < PYRAMID_LEVELS; i++)
  {
    Mat ROI;
    getRectSubPix(img_[i], sz, Point2f(x,y), ROI, CV_32FC1);
    x /= 2.0;
    y /= 2.0;
    
    // Convert to Eigen, but just the part that we want to update
    const Mat _dst(PATCH_SIZE, PATCH_SIZE, CV_64FC1, dst.data() + i * PATCH_SIZE*PATCH_SIZE, PATCH_SIZE * sizeof(double));
    if( ROI.type() == _dst.type() )
      transpose(ROI, _dst);
    else
    {
      ROI.convertTo(_dst, _dst.type());
      transpose(_dst, _dst);
    }
  }
}


void sample_pixels(Quat& qz, const Matrix2d& cov, std::vector<pixVector>& eta)
{
  (void)qz;
  (void)cov;
  (void)eta;
  
  Vector2d eta0;
  Matrix2d eta_jac;
  proj(qz, eta0, eta_jac);
  
  Eigen::SelfAdjointEigenSolver<Matrix2d> eigensolver((eta_jac.transpose() * cov * eta_jac));
  Vector2d e = eigensolver.eigenvalues();
  Matrix2d v = eigensolver.eigenvectors();
  
  double a = 2.0*std::sqrt(std::abs(e(0,0)));
  double b = 2.0*std::sqrt(std::abs(e(1,0)));
  
  for (float y = 0; y < std::round(b); y += PATCH_SIZE)
  {
    float xmax = (a/b) * std::sqrt(b*b - y*y);
    for (float x = 0; x < xmax; x += PATCH_SIZE)
    {
      float t = std::atan2(y, x);
      float ct = std::cos(t);
      float st = std::sin(t);
      if ((y*y + x*x) < (a*a*ct*ct + b*b*st*st))
      {
        Vector2d p;
        p << x, y;
        Vector2d pt = (v * p + eta0);
        if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img_[0].cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img_[0].rows)
          eta.push_back(pt);
        if (x != 0)
        {
          p(0, 0) *= -1;
          pt = (v * p + eta0);
          if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img_[0].cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img_[0].rows)
            eta.push_back(pt);
        }
        if (y != 0)
        {
          p(1, 0) *= -1;
          pt = (v * p + eta0);
          if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img_[0].cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img_[0].rows)
            eta.push_back(pt);
          if (x != 0)
          {
            p(0, 0) *= -1;
            pt = (v * p + eta0);
            if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img_[0].cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img_[0].rows)
              eta.push_back(pt);
          }
        } 
      }
    }
  }
    
}


void patch_error(const pixVector &etahat, const multiPatchVectorf &I0, multiPatchVectorf &e, multiPatchJacMatrix &J)
{
  multiPatchVectorf Ip, Im;
  multiLvlPatch(etahat, Ip);
  
  e = Ip - I0;
  Matrix2d eye2 = Matrix2d::Identity();
  
  // Perform central differencing
  for (int i = 0; i < 2; i++)
  {
    multiLvlPatch(etahat + eye2.col(i), Ip);
    multiLvlPatch(etahat - eye2.col(i), Im);
    J.col(i) = ((Ip - I0) - (Im - I0))/2.0;
  }
}

void multiLvlPatchSideBySide(multiPatchVectorf& src, multiPatchMatrixf& dst)
{
   dst = Map<multiPatchMatrixf>(src.data());
}

void multiLvlPatchToCv(multiPatchVectorf& src, Mat& dst)
{
  multiPatchMatrixf side_by_side;
  multiLvlPatchSideBySide(src, side_by_side);
  eigen2cv(side_by_side, dst);
}

void drawPatch(Mat& img, const Vector2d& pt, const Scalar& col)
{
  double alpha = 0.3;
  if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img.cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img.rows)
  {
    cv::Mat roi = img(Rect(Point(pt(0,0), pt(1,0)), Size(PATCH_SIZE, PATCH_SIZE)));
    cv::Mat color(roi.size(), CV_8UC3, col); 
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
    imwrite("/home/superjax/test/sampled.png", img);
  }
}

int main()
{  
  srand((unsigned int) time(0));
  Mat rgb = cv::imread("/home/superjax/.ros/cache/RGB00558.bmp");
  Mat grey;
  cvtColor(rgb, grey, COLOR_RGB2GRAY);
  
  // Build multilevel image
  img_[0] = grey;
  for (int i = 1; i < PYRAMID_LEVELS; i++)
  {
    pyrDown( img_[i-1], img_[i], Size( img_[i-1].cols/2, img_[i-1].rows/2 ) );
  }
  // Convert the image to a float image
  for (int i = 0; i < PYRAMID_LEVELS; i++)
  {
    img_[i].convertTo(img_[i], CV_32FC1);
  }
  
  Vector2d eta;
  eta << 253.3, 420.63827;
 
  multiPatchVectorf patch;
  
  // Extract a Multi-Level Patch
  multiLvlPatch(eta, patch);
  
  // Show the levels side by side
  multiPatchMatrixf side_by_side;
  multiLvlPatchSideBySide(patch, side_by_side);
  
  // Convert Multi level patch to cv Mat and write to file
  Mat cv_side_by_side;
  multiLvlPatchToCv(patch, cv_side_by_side);
  imwrite("/home/superjax/test/side_by_side.png", cv_side_by_side);
  
  cam_center_ << 320.0, 240.0;
  cam_F_ << 250.0, 0.0, 0.0,   0.0, 250.0, 0.0;
  
  // Plot samples
  Mat sampled;
  cvtColor(grey, sampled, COLOR_GRAY2BGR);
  
  // TEST patch_error jacobian calculation by performing Gauss-Newton Optimization on
  // a sample problem
  multiPatchJacMatrix J;
  multiPatchVectorf e;
  Vector2d etahat;
  etahat << 247.0, 412.0;
  patch_error(etahat, patch, e, J);
  
  Quat qz = Quat::Identity();
  Matrix2d cov;
  Vector2d dq;
  Matrix2d dummy;
  dq << 0.1, -0.45;
  qz = q_feat_boxplus(qz, dq);
  proj(qz, eta, cov);
  multiLvlPatch(eta, patch);
  drawPatch(sampled, eta, Scalar(0, 255, 0));
  
  // Covariance of measurement
  cov << 0.002, 0.001, 0.001, 0.004;
  Eigen::SelfAdjointEigenSolver<Matrix2d> eigensolver(cov);
  Vector2d l = eigensolver.eigenvalues();
  Matrix2d v = eigensolver.eigenvectors();
  
  // Sample from the cov
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, 1.0);
  Vector2d sample;
  sample << std::sqrt(l(0,0)) * distribution(generator),
            std::sqrt(l(1,0)) * distribution(generator);
  Quat qzhat = q_feat_boxplus(qz, v * sample);  
  proj(qzhat, etahat, dummy);
  drawPatch(sampled, etahat, Scalar(0, 0, 255));
  
  // To compare, also plot the 2 stdev patches from the qz covariance
  for (int i =0; i < 2; i++)
  {
    Vector2d p1, p2;
    proj(q_feat_boxplus(qzhat, 2.0*std::sqrt(l(i,0)) * v.col(i)), p1, dummy);
    proj(q_feat_boxplus(qzhat, -2.0*std::sqrt(l(i,0)) * v.col(i)), p2, dummy);
    drawPatch(sampled, p1, Scalar(255, 0, 0));
    drawPatch(sampled, p2, Scalar(255, 0, 0));
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  // Sample some pixels
  std::vector<pixVector> pix;
  sample_pixels(qzhat, cov, pix);
  for (int i = 0; i < pix.size(); i++)
  {
    drawPatch(sampled, pix[i], Scalar(255, 0, 255));
  }
  
  double min_error = INFINITY;
  int min_patch_idx = 0;
  for (int i = 0; i < pix.size(); i++)
  {
    std::cout << " Patch " << i << "  ====================================" << std::endl;
    int iter = 0;
    double prev_err = INFINITY;
    double current_err = e.norm();
    while (current_err > 1e-3 && iter < 100 && std::abs(1.0 - current_err/prev_err) > 0.01)
    {
      cout << "iter: " << iter << " e = " << current_err << " etahat = [" << pix[i].transpose() << "] ratio = " << std::abs(1.0 - current_err / prev_err) << std::endl;
      pix[i] = pix[i] - (J.transpose() * J).inverse() * J.transpose() * e;
      patch_error(pix[i], patch, e, J);
      iter++;
      prev_err = current_err;
      current_err = e.norm();
      
      if (current_err < min_error)
      {
        min_error = current_err;
        min_patch_idx = i;
      }
    }    
    std::cout << "iter: " << iter << " e = " << current_err << " etahat = " << pix[i].transpose() << std::endl;
    std::cout << "optimization finished\n";
  }
  drawPatch(sampled, pix[min_patch_idx], Scalar(255, 255,0));
  
  std::cout << " Optimum patch = id " << min_patch_idx << " est = [" << pix[min_patch_idx].transpose() << "] actual = [" << eta.transpose() << "]" << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  
  std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
  
  
  waitKey(0);  
}

