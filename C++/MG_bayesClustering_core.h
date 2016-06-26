//
//  MG_bayesClustering_core.h
//  bayesClusteringMixedGauss
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef __bayesClustering__MG_bayesClustering_core__
#define __bayesClustering__MG_bayesClustering_core__

#include "MG_dataUtils.h"
#include "mathUtils.h"

#include <iostream>
#define _REENTRANT
#include <cmath>

//#include <Eigen/Dense>


//using Eigen::MatrixXd;
using namespace std;

enum {
    initType_noSpecification = 0,
    // initialization method for gamma matrix
    initType_gamma_mask = 0xF,
    initType_whiteNoise = 0x1,      // randome
    initType_randomCluster = 0x2,   // randomely clustered
    initType_kMeans = 0x3,          // clustered by k-Means
    // initialization method for W0
    initType_W0_mask = 0xF00,
    initType_W0_one = 0x100,        // unit matrix
    initType_W0_hyp = 0x200,        // hyperparameter
    initType_W0_cov = 0x300,        // covariance of W0
    initType_W0_var = 0x400         // variance of diagonal W0
};


void bayesClusterMG_core(const mgCond*, const mgData*, mgResult*, mgTemp*);
void bayesMG_EStep(const mgCond*, const mgData*, mgResult*, mgTemp*);
void bayesMG_MStep(const mgCond*, const mgData*, mgResult*, mgTemp*);
double varLowerBound(const mgCond*, const mgData*, mgResult*, mgTemp*);
void setInitial(const mgCond*, const mgData*, mgResult*, mgTemp*);

#endif /* defined(__bayesClusteringPCA__MG_bayesClustering_core__) */
