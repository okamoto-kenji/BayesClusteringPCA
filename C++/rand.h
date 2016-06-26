//
//  rand.h
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.02.03.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#ifndef __bayesClusteringPCA__rand__
#define __bayesClusteringPCA__rand__

#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


void initRan();
void freeRan();
double ranInRange(double, double);
double enoise(double);
double gnoise(double);
int randomInteger(int, int);

#endif /* defined(__bayesClusteringPCA__rand__) */
