//
//  MG_dataUtils.cpp
//  bayesClusteringMixedGaussians
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "MG_dataUtils.h"

//////////  conditions
//mgCond::mgCond(){
//    kDim = 0;
//    mDim = 0;
//}

mgCond::mgCond(int _kDim){
    kDim = _kDim;
    kDimD = (double)kDim;
    
    // default settings
    minIteration = 0;
    maxIteration = 10;
    lbPosTh = 1.0e-5;
    lbNegTh = -1.0e-6;
}

mgCond::mgCond(const mgCond &other){  // do not copy
    kDim = 0;
    kDimD = 0.0;
}

mgCond::~mgCond(){
}


//////////  data
//mgData::mgData(){
//    nDim = 0;
//    lDim = 0;
//}

mgData::mgData( int _nDim, int _lDim ){
    nDim = _nDim;
    lDim = _lDim;
    nDimD = (double)nDim;
    lDimD = (double)lDim;
    dataArray = NULL; //alloc2Dmat(lDim, nDim);    // L x N
//    xn2 = NULL; //alloc1Dvec(nDim);                     // N
}

mgData::mgData(const mgData &other){  // do not copy
    nDim = 0;
    lDim = 0;
    nDimD = 0.0;
    lDimD = 0.0;
    dataArray = NULL;
//    xn2 = NULL;
}

void mgData::setDataArray(Mat2D *inDA){
    lDim = inDA->d1;
    nDim = inDA->d2;
    nDimD = (double)nDim;
    lDimD = (double)lDim;
    
    if( dataArray != NULL ){
        delete dataArray;
    }
    dataArray = new Mat2D(lDim, nDim);             // L x N
//    if( xn2 != NULL ){
//        delete xn2;
//    }
//    xn2 = new Vec1D(nDim);                              // N
    int l, n;
    for( n = 0 ; n < nDim ; n++ ){
//        xn2->p[n] = 0.0;
        for( l = 0 ; l < lDim ; l++ ){
            dataArray->p[l][n] = inDA->p[l][n];
//            xn2->p[n] += inSB->p[l][n] * inSB->p[l][n];
        }  // l
    }  // n
}

mgData::~mgData(){
    delete dataArray;
//    delete xn2;
}


//////////  results
mgResult::mgResult(int nDim, int lDim, int kDim){
    gamma = new Mat2D(nDim, kDim);          // N x K
    Nk = new Vec1D(kDim);                   // K
    Exk = new Mat2D(kDim, lDim);            // K x L
    Sk = new Ten3D(kDim, lDim, lDim);       // K x L x L
    
    // pi
    Epi = new Vec1D(kDim);                  // K
    ElnPi = new Vec1D(kDim);                // K
    alp0 = new Vec1D(kDim);                 // K
    lnCalp0 = 0.0;
    lnChatAlp = 0.0;
    alpPi = new Vec1D(kDim);                // K
    hatAlpPi = 0.0;
    
    // mu, Lambda
    Emu = new Mat2D(kDim, lDim);            // K x L
    Elmd = new Ten3D(kDim, lDim, lDim);     // K x L x L
    ElnLmd = new Vec1D(kDim);
    ElnDist = new Mat2D(kDim, nDim);        // K x N
    m0 = new Vec1D(lDim);                   // L
    beta0 = 0.0;
    W0 = new Mat2D(lDim, lDim);             // L x L
    invW0 = new Mat2D(lDim, lDim);          // L x L
    maxW0 = 0.0;
    nu0 = 0.0;
    beta_k = new Vec1D(kDim);               // K
    mk = new Mat2D(kDim, lDim);             // K x L
    Wk = new Ten3D(kDim, lDim, lDim);       // K x L x L
    maxWk = new Vec1D(kDim);                // K
    nu_k = new Vec1D(kDim);                 // K
    
    lowerBounds = new Vec1D(0);
}

mgResult::mgResult(const mgResult &other){  // do not copy
    gamma = NULL;
    Nk = NULL;
    Exk = NULL;
    Sk = NULL;
    Epi = NULL;
    ElnPi = NULL;
    alp0 = NULL;
    lnCalp0 = 0.0;
    lnChatAlp = 0.0;
    alpPi = NULL;
    hatAlpPi = 0.0;
    Emu = NULL;
    Elmd = NULL;
    ElnLmd = NULL;
    ElnDist = NULL;
    m0 = NULL;
    beta0 = 0.0;
    W0 = NULL;
    invW0 = NULL;
    maxW0 = 0.0;
    nu0 = 0.0;
    beta_k = NULL;
    mk = NULL;
    Wk = NULL;
    maxWk = NULL;
    nu_k = NULL;
    lowerBounds = NULL;
}

mgResult::~mgResult(){
    delete gamma;
    delete Nk;
    delete Exk;
    delete Sk;
    delete Epi;
    delete ElnPi;
    delete alp0;
    delete alpPi;
    delete Emu;
    delete Elmd;
    delete ElnLmd;
    delete ElnDist;
    delete m0;
    delete W0;
    delete invW0;
    delete beta_k;
    delete mk;
    delete Wk;
    delete maxWk;
    delete nu_k;
    delete lowerBounds;
}


//////////  temporary
mgTemp::mgTemp(int nDim, int lDim, int kDim){
    rho = new Vec1D(kDim);
    lnRho = new Vec1D(kDim);                      // K
    matLL = new Mat2D(lDim, lDim);
}

mgTemp::mgTemp(const mgTemp &other){  // do not copy
    rho = NULL;
    lnRho = NULL;
    matLL = NULL;
}

mgTemp::~mgTemp(){
    delete rho;
    delete lnRho;
    delete matLL;
}


//
