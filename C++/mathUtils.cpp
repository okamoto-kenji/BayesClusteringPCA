//
//  mathUtils.cpp
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "mathUtils.h"
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace Eigen;

double lngamma(double x){
    return gsl_sf_lngamma(x);
}

double digamma(double x){
    return gsl_sf_psi(x);
}


void pca(Mat2D *dat, Mat2D *res){
    int lDim = dat->d1, nDim = dat->d2, mDim = res->d2;
    double nDimD = (double)nDim;

    Vec1D mu(lDim);
    int l, m, n;
    for( l = 0 ; l < lDim ; l++ ){
        mu.p[l] = 0.0;
        for( n = 0 ; n < nDim ; n++ ){
            mu.p[l] += dat->p[l][n];
        }  // n
        mu.p[l] /= nDimD;
    }  // l
    MatrixXd cov(lDim, lDim);
    int i, j;
    double term;
    for( i = 0 ; i < lDim ; i++ ){
        for(j = 0 ; j <= i ; j++ ){
            term = 0.0;
            for( n = 0 ; n < nDim ; n++ ){
                term += (dat->p[i][n] - mu.p[i]) * (dat->p[j][n] - mu.p[j]);
            } // n
            cov(i,j) = term / nDimD;
        }  // j(i)
    }  // i(L)
    for( i = 0 ; i < lDim ; i++ ){
        for(j = 0 ; j < i ; j++ ){
            cov(j,i) = cov(i,j);
        }  // j(i)
    }  // i(L)
    SelfAdjointEigenSolver<MatrixXd> eigensolver(cov);
    VectorXd eVals = eigensolver.eigenvalues();
    MatrixXd eVecs = eigensolver.eigenvectors();

    int *mIndexes = (int *)malloc( mDim * sizeof(int) );
    int maxL = 0;
    double eV, maxEV, lastEV = INFINITY;
    for( m = 0 ; m < mDim ; m++ ){
        maxEV = -INFINITY;
        for( l = 0 ; l < lDim ; l++ ){
            eV = eVals(l);
            if( eV >= lastEV ){
                continue;
            } else {
                if( eV > maxEV ){
                    maxEV = eV;
                    maxL = l;
                }
            }
        }  // l
        for( l = 0 ; l < lDim ; l++ ){
            res->p[l][m] = eVecs(l, maxL);
        }
        lastEV = maxEV;
    }  // m

    free(mIndexes);
}


double Wishart_lnB(const Mat2D *W, const double nu, const double maxW){
    if( W->d1 != W->d2 ){
        return NAN;
    }
    int lDim = W->d1;
    double lDimD = (double)W->d1;
    int i, j;
    MatrixXd matLL(lDim, lDim);   // L x L
//    MatrixXd invMatLL(lDim, lDim);   // L x L

    double val;
    val  = - log(2.0) * (nu * lDimD / 2.0);
    val -= log(M_PI) * (lDimD * (lDimD - 1.0) / 4.0);
    for( i = 0 ; i < lDim ; i++ ){
        val -= lngamma( (nu + 1.0 - (double)i) / 2.0 );
    }  // i(L)

    for( i = 0 ; i < lDim ; i++ ){
        for( j = 0 ; j < lDim ; j++ ){
            matLL(i,j) = W->p[i][j] / maxW;
        }  // j(L)
    }  // i(L)
    val -= ( log(matLL.determinant()) + lDimD * log(maxW) ) * (nu / 2.0);
    return val;
}

double Wishart_H(const Mat2D *W, const double nu, const double ElnLmd, const double maxW){
    if( W->d1 != W->d2 ){
        return NAN;
    }
//    int lDim = W->d1;
    double lDimD = (double)W->d1;
    return (nu * lDimD - (nu - lDimD - 1.0) * ElnLmd) / 2.0 - Wishart_lnB(W, nu, maxW);
}



Vec1D::Vec1D(int _d1){
    d1 = _d1;
    if( d1 > 0 ){
        p = (double*)malloc(sizeof(double) * d1);
//        size_t i;
//        for( i = 0 ; i < d1 ; i++ ){
//            v->p[i] = 0.0;
//        }
//        memset(>p, 0, sizeof(double)*d1);
    } else {
        p = NULL;
    }
}

Vec1D::Vec1D(const Vec1D &other){
    d1 = other.d1;
    if( d1 > 0 ){
        p = (double*)malloc(sizeof(double) * d1);
        int i;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = other.p[i];
        }
    } else {
        p = NULL;
    }
}

Vec1D::~Vec1D(){
    free(p);
}

void Vec1D::resize(int d){
    if( d != d1 ){
        if( d == 0 ){
            if( p != NULL ){
                free(p);
                p = NULL;
            }
        } else {
            p = (double*)realloc( p, d * sizeof(double) );
            if( d > d1 ){
                for( int i = d1 ; i < d ; i++ ){
                    p[i] = 0.0;
                }
            }
        }
        d1 = d;
    }
}

void Vec1D::allSet(double v){
    int i;
    for( i = 0 ; i < d1 ; i++ ){
        p[i] = v;
    }
}

double Vec1D::var(){
    double n = (double)d1, mean = 0.0, ms = 0.0;
    int i;
    for( i = 0 ; i < d1 ; i++ ){
        mean += p[i];
        ms += p[i] * p[i];
    }
    mean /= n;
    ms /= n;
    return (ms - mean * mean) * n / (n - 1.0);
}


double Vec1D::sdev(){
//    double n = (double)d1, mean = 0.0, ms = 0.0;
//    int i;
//    for( i = 0 ; i < d1 ; i++ ){
//        mean += p[i];
//        ms += p[i] * p[i];
//    }
//    mean /= n;
//    ms /= n;
//    return (ms - mean * mean) * n / (n - 1.0);
    return sqrt(var());
}


void Vec1D::outputToFile(const char *filename){
    fstream fs;
    fs.open(filename, ios::out);
    int i;
    if( fs.is_open() ){
        if( d1 == 0 ){
            fs << "0" << endl;
        } else {
            for( i = 0 ; i < d1 ; i++ ){
                fs << p[i] << endl;
            }
        }
        fs.close();
    }
}

void Vec1D::outputFineToFile(const char *filename){
    fstream fs;
    fs.open(filename, ios::out);
    int i;
    char str[256];
    if( fs.is_open() ){
        if( d1 == 0 ){
            fs << "0" << endl;
        } else {
            for( i = 0 ; i < d1 ; i++ ){
                sprintf( str, "%32f", p[i] );
                fs << str << endl;
            }
        }
        fs.close();
    }
}


Mat2D::Mat2D(int _d1, int _d2){
    d1 = _d1;
    d2 = _d2;
    if( (d1 > 0) && (d2 > 0) ){
        p = (double **)malloc(sizeof(double*) * d1);
        int i;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double*)malloc(sizeof(double) * d2);
//          size_t j;
//          for( j = 0 ; j < d2 ; j++ ){
//              m->p[i][j] = 0.0;
//          }
//            memset(m->p, 0, sizeof(double)*d1*d2);
        }
    } else {
        p = NULL;
    }
}

Mat2D::Mat2D(const Mat2D &other){
    d1 = other.d1;
    d2 = other.d2;
    if( (d1 > 0) && (d2 > 0) ){
        p = (double **)malloc(sizeof(double*) * d1);
        int i, j;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double*)malloc(sizeof(double) * d2);
            for( j = 0 ; j < d2 ; j++ ){
                p[i][j] = other.p[i][j];
            }
        }
    } else {
        p = NULL;
    }
}

Mat2D::~Mat2D(){
    if( p != NULL ){
        int i;
        for( i = 0 ; i < d1 ; i++ ){
            free(p[i]);
        }
        free(p);
    }
}

//void Mat2D::resize(int _d1, int _d2){
//    int i, j;
//    if( d1_ == 0 ){
//        for( i = 0 ; i < d1 ; i++ ){
//            free(p[i]);
//        }
//        free(p);
//    } else {
//        p = (double*)realloc( p, sizeof(double)*d1 );
//        if( d > d1 ){
//            for( int i = d1 ; i < d ; i++ ){
//                p[i] = 0.0;
//            }
//        }
//    }
//    d1 = _d1;
//    d2 = _d2;
//}

void Mat2D::allSet(double v){
    int i, j;
    for( i = 0 ; i < d1 ; i++ ){
        for( j = 0 ; j< d2 ; j++ ){
            p[i][j] = v;
        }
    }
}


Ten3D::Ten3D(int _d1, int _d2, int _d3){
    d1 = _d1;
    d2 = _d2;
    d3 = _d3;
    if( (d1 > 0) && (d2 > 0) && (d3 > 0) ){
        p = (double ***)malloc(sizeof(double**) * d1);
        int i, j;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double**)malloc(sizeof(double*) * d2);
            for( j = 0 ; j < d2 ; j++ ){
                p[i][j] = (double*)malloc(sizeof(double) * d3);
//              size_t k;
//              for( k = 0 ; k < d3 ; k++ ){
//                  m->p[i][j][k] = 0.0;
//              }
            }
        }
    } else {
        p = NULL;
    }
//    memset(t->p, 0, sizeof(double)*d1*d2*d3);
}

Ten3D::Ten3D(const Ten3D &other){
    d1 = other.d1;
    d2 = other.d2;
    d3 = other.d3;
    if( (d1 > 0) && (d2 > 0) && (d3 > 0) ){
        p = (double ***)malloc(sizeof(double**) * d1);
        int i, j, k;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double**)malloc(sizeof(double*) * d2);
            for( j = 0 ; j < d2 ; j++ ){
                p[i][j] = (double*)malloc(sizeof(double) * d3);
                for( k = 0 ; k < d3 ; k++ ){
                    p[i][j][k] = other.p[i][j][k];
                }
            }
        }
    } else {
        p = NULL;
    }
}

Ten3D::~Ten3D(){
    if( p != NULL ){
        int i, j;
        for( i = 0 ; i < d1 ; i++ ){
            for( j = 0 ; j < d2 ; j++ ){
                free(p[i][j]);
            }
            free(p[i]);
        }
        free(p);
    }
}


Ten4D::Ten4D(int _d1, int _d2, int _d3, int _d4){
    d1 = _d1;
    d2 = _d2;
    d3 = _d3;
    d4 = _d4;
    if( (d1 > 0) && (d2 > 0) && (d3 > 0) && (d4 > 0) ){
        p = (double ****)malloc(sizeof(double***) * d1);
        int i, j, k;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double***)malloc(sizeof(double**) * d2);
            for( j = 0 ; j < d2 ; j++ ){
                p[i][j] = (double**)malloc(sizeof(double*) * d3);
                for( k = 0 ; k < d3 ; k++ ){
                    p[i][j][k] = (double*)malloc(sizeof(double) * d4);
//                  size_t l;
//                  for( l = 0 ; l < d4 ; l++ ){
//                      m->p[i][j][k][l] = 0.0;
//                  }
//                  memset(t->p, 0, sizeof(double)*d1*d2*d3*d4);
                }
            }
        }
    } else {
        p = NULL;
    }
}

Ten4D::Ten4D(const Ten4D &other){
    d1 = other.d1;
    d2 = other.d2;
    d3 = other.d3;
    d4 = other.d4;
    if( (d1 > 0) && (d2 > 0) && (d3 > 0) && (d4 > 0) ){
        p = (double ****)malloc(sizeof(double***) * d1);
        int i, j, k, l;
        for( i = 0 ; i < d1 ; i++ ){
            p[i] = (double***)malloc(sizeof(double**) * d2);
            for( j = 0 ; j < d2 ; j++ ){
                p[i][j] = (double**)malloc(sizeof(double*) * d3);
                for( k = 0 ; k < d3 ; k++ ){
                    p[i][j][k] = (double*)malloc(sizeof(double) * d4);
                    for( l = 0 ; l < d4 ; l++ ){
                      p[i][j][k][l] = other.p[i][j][k][l];
                    }
                }
            }
        }
    } else {
        p = NULL;
    }
}

Ten4D::~Ten4D(){
    if( p != NULL ){
        int i, j, k;
        for( i = 0 ; i < d1 ; i++ ){
            for( j = 0 ; j < d2 ; j++ ){
                for( k = 0 ; k < d3 ; k++ ){
                    free(p[i][j][k]);
                }
                free(p[i][j]);
            }
            free(p[i]);
        }
        free(p);
    }
}

//
