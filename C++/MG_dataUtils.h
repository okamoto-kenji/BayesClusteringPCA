//
//  MG_dataUtils.h
//  bayesClusteringMixGauss
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef __bayesClustering__MG_dataUtils__
#define __bayesClustering__MG_dataUtils__

#include <stdio.h>
#include <string>
#include "mathUtils.h"

using namespace std;

class mgCond {
public:
    int kDim;
    double kDimD;
    int minIteration;
    int maxIteration;
    double lbPosTh;     // positive threshold for lower bound
    double lbNegTh;     // negative threshold (negative value) for lower bound
    int initType;
    
//    mgCond();
    mgCond(int _kDim);
    mgCond(const mgCond&);
    ~mgCond();
};

class mgData{
public:
    int nDim, lDim;
    double nDimD, lDimD;
    Mat2D *dataArray;      // L x N
//    Vec1D *xn2;                 // N
    
//    mgData();
    mgData(int _nDim = 0, int _lDim = 0);
    mgData(const mgData&);
    ~mgData();
    
    int load(string);
    void setDataArray(Mat2D*);
};

class mgResult{
public:
    // Z, gamma
    Mat2D *gamma;               // N x K
//    Ten3D *Ez;                  // K x N x M
    Vec1D *Nk;                  // K
    Mat2D *Exk;                 // K x L
    Ten3D *Sk;                  // K x L x L
    
    // pi
    Vec1D *Epi;                 // K
    Vec1D *ElnPi;               // K
    Vec1D *alp0;                // K
    double lnCalp0, lnChatAlp;
    Vec1D *alpPi;               // K
    double hatAlpPi;

    // mu, Lambda
    Mat2D *Emu;                 // K x L
    Ten3D *Elmd;                // K x L x L
    Vec1D *ElnLmd;              // K
    Mat2D *ElnDist;              // K x N
    Vec1D *m0;                  // L
    double beta0;
    Mat2D *W0, *invW0;          // L x L
    double maxW0;
    double nu0;
    Vec1D *beta_k;              // K
    Mat2D *mk;                  // K x L
    Ten3D *Wk;                  // K x L x L
    Vec1D *maxWk;               // K
    Vec1D *nu_k;                // K
    
    Vec1D *lowerBounds;         // flexible
    int iterCnt;                   // iteration count
    
    mgResult(int nDim, int lDim, int kDim);
    mgResult(const mgResult&);
    ~mgResult();
    
    void outputToFile(const char*, const mgCond*, const mgData*, int);
};

class mgTemp{
public:
    Vec1D *rho, *lnRho;         // K
    Mat2D *matLL;               // L x L
    
    mgTemp(int nDim, int lDim, int kDim);
    mgTemp(const mgTemp&);
    ~mgTemp();
};


#endif /* defined(__bayesClustering__MG_dataUtils__) */

