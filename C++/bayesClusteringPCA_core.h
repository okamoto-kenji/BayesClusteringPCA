//
//  bayesClusteringPCA_core.h
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.02.03.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef bayesClusteringPCA_bayesClusteringPCA_core_h
#define bayesClusteringPCA_bayesClusteringPCA_core_h

#include "dataUtils.h"
#include "mathUtils.h"

#include <iostream>
#define _REENTRANT
#include <cmath>

//#include <Eigen/Dense>


//using Eigen::MatrixXd;
using namespace std;

enum {
    initType_noSpecification = 0,
    initType_whiteNoise = 1,
    initType_randomCluster = 2,
    initType_kMeans = 3
};


void bayesClusterPCA_core(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*);
void bayesPcaEStep(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*);
double ElnDistance(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*, int, int);
void bayesPcaMStep(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*);
double varLowerBound(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*);
void setInitial(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*);
void setParamsFromGamma(const pcaCond*, const pcaData*, pcaResult*, pcaTemp*, double);

#endif
