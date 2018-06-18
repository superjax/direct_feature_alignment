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
#include <opencv2/aruco.hpp>  
#include <iostream>
#include <random>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace quat;

#define PATCH_SIZE 6
#define PYRAMID_LEVELS 2

#define TIMING

typedef Matrix<double, 6, 1> uVector;
typedef Matrix<double, 7, 1> eVector;
typedef Matrix<double, 4, 1> zVector;
typedef Matrix<float, 2, 1> pixVector;
typedef Matrix<uint8_t, PATCH_SIZE,PATCH_SIZE> patchMat;
typedef Matrix<uint8_t, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 1> multiPatchVectorI;
typedef Matrix<float, PATCH_SIZE, PATCH_SIZE*PYRAMID_LEVELS> multiPatchMatrixf;
typedef Matrix<float, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 1> multiPatchVectorf;
typedef Matrix<float, PATCH_SIZE*PATCH_SIZE*PYRAMID_LEVELS, 2> multiPatchJacMatrix;

Mat img_[PYRAMID_LEVELS];

Vector2d cam_center_;
Matrix<double, 2, 3> cam_F_;

void proj(Quat qz, Vector2f& eta, Matrix2f& jac)
{
  Vector3d zeta = qz.rota(e_z);
  Matrix3d sk_zeta = skew(zeta);
  double ezT_zeta = e_z.transpose() * zeta;
  MatrixXd T_z = T_zeta(qz);
  
  eta = (cam_F_ * zeta / ezT_zeta + cam_center_).cast<float>();
  jac = (-cam_F_ * ((sk_zeta * T_z)/ezT_zeta - (zeta * e_z.transpose() * sk_zeta * T_z)/(ezT_zeta*ezT_zeta))).cast<float>();
}

void multiLvlPatch(const pixVector &eta, multiPatchVectorf& dst)
{
  Size sz(PATCH_SIZE, PATCH_SIZE);
  float x = eta(0,0);
  float y = eta(1,0);
  for (int i = 0; i < PYRAMID_LEVELS; i++)
  {
    const Mat _dst(PATCH_SIZE, PATCH_SIZE, CV_32FC1, dst.data() + i * PATCH_SIZE*PATCH_SIZE, PATCH_SIZE * sizeof(float));
    getRectSubPix(img_[i], sz, Point2f(x,y), _dst);
    x /= 2.0;
    y /= 2.0;
  }
}

Eigen::SelfAdjointEigenSolver<Matrix2f> eigensolver;
void sample_pixels(Quat& qz, const Matrix2f& cov, std::vector<Vector2f>& eta)
{  
  Vector2f eta0;
  Matrix2f eta_jac;
  proj(qz, eta0, eta_jac);
  
  Matrix2f P = eta_jac * cov * eta_jac.transpose();
  eigensolver.computeDirect(P);
  Vector2f e = eigensolver.eigenvalues();
  Matrix2f v = eigensolver.eigenvectors();
  double a = 3.0*std::sqrt(std::abs(e(0,0)));
  double b = 3.0*std::sqrt(std::abs(e(1,0)));
  
  /// TODO: Spiral out from center, instead of along axis to increase search efficiency later
  Vector2f p, pt;
  for (float y = 0; y < b; y += PATCH_SIZE * 1.0)
  {
    p(1,0) = y;
    float xmax = (a/b) * std::sqrt(b*b - y*y);
    for (float x = 0; x < xmax; x += PATCH_SIZE * 1.0)
    {
      p(0,0) = x;
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

multiPatchVectorf Ip, Im;
Matrix2f eye2 = Matrix2f::Identity();
void patch_error(const pixVector &etahat, const multiPatchVectorf &I0, multiPatchVectorf &e, multiPatchJacMatrix &J)
{
  multiLvlPatch(etahat, Ip);
  
  e = Ip - I0;  
  
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

void drawPatch(Mat& img, const Vector2f& pt, const Scalar& col, double alpha=0.3)
{
  if (pt(0,0) > PATCH_SIZE && pt(0,0) + PATCH_SIZE < img.cols && pt(1,0) > PATCH_SIZE && pt(1,0) + PATCH_SIZE < img.rows)
  {
    cv::Mat roi = img(Rect(Point(pt(0,0), pt(1,0)), Size(PATCH_SIZE, PATCH_SIZE)));
    cv::Mat color(roi.size(), CV_8UC3, col); 
    cv::addWeighted(color, alpha, roi, 1.0 - alpha , 0.0, roi); 
  }
}

int main()
{  
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
  
  // Make up an intrinsic matrix
  cam_center_ << 320.0, 240.0;
  cam_F_ << 250.0, 0.0, 0.0,   0.0, 250.0, 0.0;
  
  std::default_random_engine generator;
  generator.seed(time(0));
  std::normal_distribution<double> distribution(0, 1.0);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  
  std::vector<pixVector> pix;
  std::vector<pixVector> pix_copy;
  pix.reserve(100);
  pix_copy.reserve(100);
  
  int timitsum = 0;
  int timitcount = 0;
  int successcount =0;
  for (int timit = 0; timit < 100; timit++)
  {
    // Plot samples
    Mat sampled;
    cvtColor(grey, sampled, COLOR_GRAY2BGR);
    
    // Test patch_error jacobian calculation by performing Gauss-Newton Optimization on
    // a sample problem
    multiPatchJacMatrix J;
    multiPatchVectorf e;
    Vector2f etahat;
    
    Quat qz = Quat::Identity();
    Matrix2f cov;
    Vector2d dq;
    Matrix2f dummy;
    Vector2f eta;
    multiPatchVectorf patch;
    dq << 0.5*uniform(generator), 0.4*uniform(generator);
    qz = q_feat_boxplus(qz, dq);
    proj(qz, eta, cov);
    
    // Sample the patch
    multiLvlPatch(eta, patch);
    
    // Draw the patch side-by-side
    //    Mat side_by_side;
    //    multiLvlPatchToCv(patch, side_by_side);
    //    imwrite("side-by-side.png", side_by_side);
    
    // Covariance of measurement
    float dx = (0.002 * uniform(generator)) + 0.0025;
    float dy = (0.002 * uniform(generator)) + 0.0025;
    float dxdy = (0.001 * uniform(generator));
    cov << dx, dxdy, dxdy, dy;
    eigensolver.computeDirect(cov);
    Vector2f l = eigensolver.eigenvalues();
    Matrix2f v = eigensolver.eigenvectors();
    
    // Sample from the cov
    Vector2f sample;
    sample << std::sqrt(std::abs(l(0,0))) * distribution(generator),
              std::sqrt(std::abs(l(1,0))) * distribution(generator);
    Quat qzhat = q_feat_boxplus(qz, (v * sample).cast<double>());  
    proj(qzhat, etahat, dummy);
    
    if (etahat(0,0) < PATCH_SIZE || etahat(0,0) + PATCH_SIZE > img_[0].cols || etahat(1,0) < PATCH_SIZE || etahat(1,0) + PATCH_SIZE > img_[0].rows)
      continue;
    

    pix.clear();
    pix_copy.clear();
    auto start = std::chrono::high_resolution_clock::now();
    // Sample some pixels
    sample_pixels(qzhat, cov, pix);
    
    int min_patch_idx = 0;
    double tol = 1;
    for (int i = 0; i < pix.size(); i++)
    {
      pix_copy.push_back(pix[i]);
      int iter = 0;
      bool done = false;
      double prev_err = INFINITY;
      double current_err = INFINITY;
      do
      {
        patch_error(pix[i], patch, e, J);
        pix[i] = pix[i] - (J.transpose() * J).inverse() * J.transpose() * e;
        
        // Make sure the optimization isn't doing anything stupid
        if (pix[i](0,0) < PATCH_SIZE || pix[i](0,0) + PATCH_SIZE > img_[0].cols || pix[i](1,0) < PATCH_SIZE || pix[i](1,0) + PATCH_SIZE > img_[0].rows)
          break;
        if ((pix[i] - pix_copy[i]).norm() > PATCH_SIZE)
          break;       
        
        iter++;
        prev_err = current_err;
        current_err = e.norm();
        
        if (current_err < tol)
        {
          done = true;
          min_patch_idx = i;
        }
      } while (!done && iter < 25 && std::abs(1.0 - current_err/prev_err) > 0.05);
      if (done)
        break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    int us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Call it a success if we matched to within 1 pixel
    double actual_error = (pix[min_patch_idx] - eta).norm();    
    std::cout << timit << ((actual_error < 1) ? " success - " : " failed - ") << "patch: " << min_patch_idx << " err: " << actual_error 
              << ", took " << us << "us" << " cov = [" << cov(0,0) << ", " << cov(0,1) << "; " << cov(1,0) << ", " << cov(1,1) << "]" << std::endl;
    
    if (actual_error < 1)
      successcount++;
    else
    {
    }

    for (int i = 0; i < pix_copy.size(); i++)
      drawPatch(sampled, pix_copy[i], Scalar(255, 0, 0)); // original patch locations (blue)
    for (int i = 0; i < pix.size(); i++)
      drawPatch(sampled, pix[i], Scalar(255, 0, 255)); // where the patches went (magenta)

    // To compare, also plot the 2 stdev patches from the qz covariance
    for (int i =0; i < 2; i++)
    {
      Vector2f p1, p2;
      proj(q_feat_boxplus(qzhat, 2.0*std::sqrt(l(i,0)) * v.cast<double>().col(i)), p1, dummy);
      proj(q_feat_boxplus(qzhat, -2.0*std::sqrt(l(i,0)) * v.cast<double>().col(i)), p2, dummy);
      drawPatch(sampled, p1, Scalar(0, 0, 255)); // qz distribution (red)
      drawPatch(sampled, p2, Scalar(0, 0, 255));
    } 

    drawPatch(sampled, pix[min_patch_idx], Scalar(0, 255, 255), 0.8); // optimized guess (yellow)
    drawPatch(sampled, eta, Scalar(0, 255, 0), 0.5); // (actual answer (green))

    
    imwrite("/home/superjax/test/sampled" + std::to_string(timit) + ".png", sampled);
    
    timitcount++;
    timitsum += us;
  }
  std::cout << "average " << ((double) timitsum / (double) timitcount)*1e-6 << " seconds, success = " << (double)successcount / (double)timitcount << std::endl;
   
  
  waitKey(0);  
}

