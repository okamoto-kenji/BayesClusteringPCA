//
//  mathUtils.h
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef __bayesClusteringPCA__mathUtils__
#define __bayesClusteringPCA__mathUtils__

#include "rand.h"

//#include <Eigen/Dense>
#include <cstdlib>
#include <cstring>
#include <cmath>

//using namespace Eigen;


class Vec1D {
public:
    int d1;
    double *p;

    Vec1D(int _d1=0);
    Vec1D(const Vec1D&);
    ~Vec1D();

    void resize(int);
    void allSet(double);
    double var();
    double sdev();
    void outputToFile(const char*);
    void outputFineToFile(const char*);
};

class Mat2D {
public:
    int d1, d2;
    double **p;

    Mat2D(int _d1=0, int _d2=0);
    Mat2D(const Mat2D&);
    ~Mat2D();

//    void resize(int, int);
    void allSet(double);
};

class Ten3D {
public:
    int d1, d2, d3;
    double ***p;
    
    Ten3D(int _d1=0, int _d2=0, int _d3=0);
    Ten3D(const Ten3D&);
    ~Ten3D();

//    void resize(int, int, int);
};

class Ten4D {
public:
    int d1, d2, d3, d4;
    double ****p;
    
    Ten4D(int _d1=0, int _d2=0, int _d3=0, int _d4=0);
    Ten4D(const Ten4D&);
    ~Ten4D();

//    void resize(int, int, int, int);
};


double lngamma(double);
double digamma(double);
void pca(Mat2D*, Mat2D*);
double sdev(Vec1D*);

double Wishart_lnB(const Mat2D *W, const double nu, const double maxW);
double Wishart_H(const Mat2D *W, const double nu, const double ElnLmd, const double maxW);

#endif /* defined(__bayesClusteringPCA__mathUtils__) */
