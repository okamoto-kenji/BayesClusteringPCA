//
//  dataUtils.cpp
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "dataUtils.h"


//////////  conditions
//pcaCond::pcaCond(){
//    kDim = 0;
//    mDim = 0;
//}

pcaCond::pcaCond(int _kDim, int _mDim){
    kDim = _kDim;
    mDim = _mDim;
    kDimD = (double)kDim;
    mDimD = (double)mDim;

    // default settings
    minIteration = 0;
    maxIteration = 10;
    lbPosTh = 1.0e-5;
//    lbNegTh = -1.0e-5;

    // simulated annealing
    annealSteps = 0;
    maxTFlucAmp = 0.25;
    minTFlucAmp = 1.0e-4;
    //
}

pcaCond::pcaCond(const pcaCond &other){  // do not copy
    kDim = 0;
    mDim = 0;
    kDimD = 0.0;
    mDimD = 0.0;
}

pcaCond::~pcaCond(){
}


//////////  data
//pcaData::pcaData(){
//    nDim = 0;
//    lDim = 0;
//}

pcaData::pcaData( int _nDim, int _lDim ){
    nDim = _nDim;
    lDim = _lDim;
    nDimD = (double)nDim;
    lDimD = (double)lDim;
    spectrumBundle = NULL;      // L x N
    xn2 = NULL;                 // N
    baseGamma = NULL;           // N x K
}

pcaData::pcaData(const pcaData &other){  // do not copy
    nDim = 0;
    lDim = 0;
    nDimD = 0.0;
    lDimD = 0.0;
    spectrumBundle = NULL;
    xn2 = NULL;
    baseGamma = NULL;
}

void pcaData::setSpectrumBundle(Mat2D *inSB){
    lDim = inSB->d1;
    nDim = inSB->d2;
    nDimD = (double)nDim;
    lDimD = (double)lDim;

    if( spectrumBundle != NULL ){
        delete spectrumBundle;
    }
    spectrumBundle = new Mat2D(lDim, nDim);             // L x N
    if( xn2 != NULL ){
        delete xn2;
    }
    xn2 = new Vec1D(nDim);                              // N
    int l, n;
    for( n = 0 ; n < nDim ; n++ ){
        xn2->p[n] = 0.0;
        for( l = 0 ; l < lDim ; l++ ){
            spectrumBundle->p[l][n] = inSB->p[l][n];
            xn2->p[n] += inSB->p[l][n] * inSB->p[l][n];
        }  // l
    }  // n
}

pcaData::~pcaData(){
    delete spectrumBundle;
    delete xn2;
    delete baseGamma;
}


//////////  results
pcaResult::pcaResult(int nDim, int lDim, int kDim, int mDim){
    gamma = new Mat2D(nDim, kDim);                 // N x K
    Ez = new Ten3D(kDim, nDim, mDim);              // K x N x M
    EzzT = new Ten4D(kDim, nDim, mDim, mDim);      // K x N x M x M
    Ez2 = new Mat2D(kDim, nDim);                   // K x N
    
    Epi = new Vec1D(kDim);                         // K
    ElnPi = new Vec1D(kDim);                       // K
    Emu = new Mat2D(kDim, lDim);                   // K x L
    Emu2 = new Vec1D(kDim);                        // K
    Ew = new Ten3D(kDim, lDim, mDim);              // K x L x M
    EwTw = new Ten3D(kDim, mDim, mDim);            // K x M x M
    Ewm2 = new Mat2D(kDim, mDim);                  // K x M
    Ealp = new Mat2D(kDim, mDim);
    ElnAlp = new Mat2D(kDim, mDim);                // K x M
    Elmd = new Vec1D(kDim);
    ElnLmd = new Vec1D(kDim);                      // K
    
    alp0 = new Vec1D(kDim);                        // K
    lnCalp0 = 0.0;
    lnChatAlp = 0.0;
    betaMu = new Vec1D(kDim);
    aAlp = new Vec1D(kDim);
    bAlp = new Vec1D(kDim);                        // K
    aLmd = new Vec1D(kDim);
    bLmd = new Vec1D(kDim);                        // K
    
    Nk = new Vec1D(kDim);                          // K
    mZeta = new Ten3D(kDim, nDim, mDim);           // K x N x M
    sigZeta = new Ten4D(kDim, nDim, mDim, mDim);   // K x N x M x M
    mMu = new Mat2D(kDim, lDim);                   // K x L
    sigMu_1 = new Vec1D(kDim);                     // K (x L x L)
    mW = new Ten3D(kDim, lDim, mDim);              // K x L x M
    sigW = new Ten3D(kDim, mDim, mDim);            // K x M x M
    tldAalp = new Vec1D(kDim);                     // K
    tldBalp = new Mat2D(kDim, mDim);               // K x M
    tldAlmd = new Vec1D(kDim);
    tldBlmd = new Vec1D(kDim);                     // K
    alpPi = new Vec1D(kDim);                       // K
    hatAlpPi = 0.0;
    ElnDist = new Mat2D(kDim, nDim);               // K x N
    
    lowerBounds = new Vec1D(0);

    // simulated annealing
    tFlucAmp = 0.0;
    //
}

pcaResult::pcaResult(const pcaResult &other){  // do not copy
    gamma = NULL;
    Ez = NULL;
    EzzT = NULL;
    Ez2 = NULL;
    Epi = NULL;
    ElnPi = NULL;
    Emu = NULL;
    Emu2 = NULL;
    Ew = NULL;
    EwTw = NULL;
    Ewm2 = NULL;
    Ealp = NULL;
    ElnAlp = NULL;
    Elmd = NULL;
    ElnLmd = NULL;
    alp0 = NULL;
    lnCalp0 = 0.0;
    lnChatAlp = 0.0;
    betaMu = NULL;
    aAlp = NULL;
    bAlp = NULL;
    aLmd = NULL;
    bLmd = NULL;
    Nk = NULL;
    mZeta = NULL;
    sigZeta = NULL;
    mMu = NULL;
    sigMu_1 = NULL;
    mW = NULL;
    sigW = NULL;
    tldAalp = NULL;
    tldBalp = NULL;
    tldAlmd = NULL;
    tldBlmd = NULL;
    alpPi = NULL;
    hatAlpPi = 0.0;
    ElnDist = NULL;
    lowerBounds = NULL;

    // simulated annealing
    tFlucAmp = 0.0;
    //
}

pcaResult::~pcaResult(){
    delete gamma;
    delete Ez;
    delete EzzT;
    delete Ez2;
    delete Epi;
    delete ElnPi;
    delete Emu;
    delete Emu2;
    delete Ew;
    delete EwTw;
    delete Ewm2;
    delete Ealp;
    delete ElnAlp;
    delete Elmd;
    delete ElnLmd;
    delete alp0;
    delete betaMu;
    delete aAlp;
    delete bAlp;
    delete aLmd;
    delete bLmd;
    delete Nk;
    delete mZeta;
    delete sigZeta;
    delete mMu;
    delete sigMu_1;
    delete mW;
    delete sigW;
    delete tldAalp;
    delete tldBalp;
    delete tldAlmd;
    delete tldBlmd;
    delete alpPi;
    delete ElnDist;
    delete lowerBounds;
}


//////////  temporary
pcaTemp::pcaTemp(int nDim, int lDim, int kDim, int mDim){
    rho = new Vec1D(kDim);
    lnRho = new Vec1D(kDim);                      // K
    vecM = new Vec1D(mDim);                       // M

//    matMM = new MatrixXd(mDim,mDim);
//    matML = new MatrixXd(mDim, lDim);
}

pcaTemp::pcaTemp(const pcaTemp &other){  // do not copy
    rho = NULL;
    lnRho = NULL;
    vecM = NULL;
//    matMM = NULL;
//    matML = NULL;
}

pcaTemp::~pcaTemp(){
    delete rho;
    delete lnRho;
    delete vecM;
//    delete matMM;
//    delete matML;
}


//
