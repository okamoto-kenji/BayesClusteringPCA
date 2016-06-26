//
//  dataUtils.h
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef __bayesClusteringPCA__dataUtils__
#define __bayesClusteringPCA__dataUtils__

#include <stdio.h>
#include <string>
#include "mathUtils.h"

using namespace std;

class pcaCond {
public:
    int kDim, mDim;
    double kDimD, mDimD;
    int minIteration;
    int maxIteration;
    double lbPosTh;     // positive threshold for lower bound
    double lbNegTh;     // negative threshold (negative value) for lower bound
    int initType;

    // simulated annealing
    int annealSteps;
    double maxTFlucAmp, minTFlucAmp;
    //

//    pcaCond();
    pcaCond(int _kDim, int _mDim);
    pcaCond(const pcaCond&);
    ~pcaCond();
};

class pcaData{
public:
    int nDim, lDim;
    double nDimD, lDimD;
    Mat2D *spectrumBundle;      // L x N
    Vec1D *xn2;                 // N
    Mat2D *baseGamma;           // N x K

//    pcaData();
    pcaData(int _nDim = 0, int _lDim = 0);
    pcaData(const pcaData&);
    ~pcaData();

    int load(string, string);
    void setSpectrumBundle(Mat2D*);
};

class pcaResult{
public:
    Mat2D *gamma;               // N x K
    Ten3D *Ez;                  // K x N x M
    Ten4D *EzzT;                // K x N x M x M
    Mat2D *Ez2;                 // K x N
    
    Vec1D *Epi, *ElnPi;         // K
    Mat2D *Emu;                 // K x L
    Vec1D *Emu2;                // K
    Ten3D *Ew;                  // K x L x M
    Ten3D *EwTw;                // K x M x M
    Mat2D *Ewm2;                // K x M
    Mat2D *Ealp, *ElnAlp;       // K x M
    Vec1D *Elmd, *ElnLmd;       // K
    
    Vec1D *alp0;                // K
    double lnCalp0, lnChatAlp;
    Vec1D *betaMu, *aAlp, *bAlp;// K
    Vec1D *aLmd, *bLmd;         // K
    
    Vec1D *Nk;                  // K
    Ten3D *mZeta;               // K x N x M
    Ten4D *sigZeta;             // K x N x M x M
    Mat2D *mMu;                 // K x L
    Vec1D *sigMu_1;             // K (x L x L)
    Ten3D *mW;                  // K x L x M
    Ten3D *sigW;                // K x M x M
    Vec1D *tldAalp;             // K
    Mat2D *tldBalp;             // K x M
    Vec1D *tldAlmd, *tldBlmd;   // K
    Vec1D *alpPi;               // K
    double hatAlpPi;
    Mat2D *ElnDist;              // K x N

    // simulated annealing
    double tFlucAmp;
    //

    Vec1D *lowerBounds;         // flexible
    int iterCnt;                   // iteration count

    pcaResult(int nDim, int lDim, int kDim, int mDim);
    pcaResult(const pcaResult&);
    ~pcaResult();

    void outputToFile(const char*, const pcaCond*, const pcaData*, int);
};

class pcaTemp{
public:
    Vec1D *rho, *lnRho;         // K
    Vec1D *vecM;                 // M
//    MatrixXd *matMM, *invMatMM;   // M x M
//    MatrixXd *matML;             // M x L

    pcaTemp(int nDim, int lDim, int kDim, int mDim);
    pcaTemp(const pcaTemp&);
    ~pcaTemp();
};


#endif /* defined(__bayesClusteringPCA__dataUtils__) */


//
