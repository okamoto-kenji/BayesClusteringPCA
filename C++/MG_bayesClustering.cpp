//
//  MG_bayesClustering.cpp
//  bayesClusteringMixedGauss
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "MG_bayesClustering.h"
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, const char * argv[]){
    
    char  in_name[255], out_name[255];  //, logFilename[255];
    time_t  startTime = time((time_t *)NULL);
    
    int kDim = 0;
    int maxIteration = 0, rep;
    double lbPosTh = 0.0;
    
#ifdef _OPENMP
    cout << "OpenMP defined." << endl;
#endif
    
    /*******************************
     *  Get command line arguments  *
     *******************************/
    if (argc != 6) {
        // No file name entered in the command line
        cout << "\nVariational-Bayese Clustering of Mixed Gaussians" << endl;
        cout << "    built on 2015.03.04" << endl;
        cout << "Syntax : bcMixGauss kDim maxIteration lbPosTh repetition filename" << endl;
        cout << "         kDim         : k-dimension, maximum number of cluster" << endl;
        cout << "         maxIteration : maximum iteration if inference does not reach threshold" << endl;
        cout << "         lbPosTh      : stop iteration if inference reaches this value" << endl;
        cout << "         repetition   : number of repetition with an initial condition" << endl;
        cout << "         filename     : name of data file" << endl;
        cout << "Example: bcMixGauss 5 1000 1e-10 100 data" << endl << endl;
        exit(1);
    } else if( argc == 6 ){
        // Set parameters
        kDim = atoi(argv[1]);
        maxIteration = atoi(argv[2]);
        lbPosTh = atof(argv[3]);
        rep = atoi(argv[4]);
        
        // Set the output filename
        strncpy( in_name, argv[5], sizeof(in_name) );
        
        strncpy( out_name, in_name, sizeof(out_name) );
        strncat( out_name, ".lowerBounds", sizeof(out_name) - strlen(out_name) - 1 );
        
        //        strncpy( logFilename, out_name, sizeof(logFilename) );
        //        strncat( logFilename, ".log", sizeof(logFilename) - strlen(logFilename) - 1 );
        
        cout << " kDim: " << kDim << endl;
        cout << " maxIteration: " << maxIteration << endl;
        cout << " lbPosTh: " << lbPosTh << endl;
        cout << " in_name: " << in_name << endl;
        cout << " out_name: " << out_name << endl;
        //        cout << " logFilename: " << logFilename << endl;
        //        logFP = fopen( logFilename, "w" );
    }
    /*******************************
     *          Print Title         *
     *******************************/
    initRan();
    cout << "*****************************************************************" << endl;
    cout << "*       Variational-Bayese Clustering of Mixed Gaussians        *" << endl;
    cout << "*                             by                                *" << endl;
    cout << "*                        OKAMOTO Kenji                          *" << endl;
    cout << "*                            RIKEN                              *" << endl;
    cout << "*****************************************************************" << endl;
    cout << "v 0.01-alpha - built on 2015.03.04" << endl << endl;
    
#ifdef _OPENMP
    cout << "The number of processors is " << omp_get_num_procs() << endl << endl;
#endif
    
    int minIteration = 0;
//    double lbNegTh = -1.0e-9;
    
    int r;
    
    mgData *data = new mgData();
    data->load(in_name);
    int nDim = data->nDim, lDim = data->lDim;
    
    mgCond **conds = (mgCond**)malloc( rep * sizeof(mgCond*) );
    mgResult **results = (mgResult**)malloc( rep * sizeof(mgResult*) );
    mgTemp **temps = (mgTemp**)malloc( rep * sizeof(mgTemp*) );
    Vec1D scores(rep);
    scores.allSet(NAN);
    
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for( r = 0 ; r < rep ; r++ ){
        
        conds[r] = new mgCond(kDim);
        conds[r]->minIteration = minIteration;
        conds[r]->maxIteration = maxIteration;
        conds[r]->lbPosTh = lbPosTh;
        conds[r]->lbNegTh = -0.1 * lbPosTh;
        
        if( r < (rep/4) ){
            conds[r]->initType = initType_W0_one;
            if( r < (rep/8) ){
                conds[r]->initType |= initType_whiteNoise;
            } else {
                conds[r]->initType |= initType_kMeans;
            }
        } else if( r < (rep/2) ){
            conds[r]->initType = initType_W0_hyp;
            if( r < (rep*3/8) ){
                conds[r]->initType |= initType_whiteNoise;
            } else {
                conds[r]->initType |= initType_kMeans;
            }
        } else if( r < (rep*3/4) ){
            conds[r]->initType = initType_W0_var;
            if( r < (rep*5/8) ){
                conds[r]->initType |= initType_whiteNoise;
            } else {
                conds[r]->initType |= initType_kMeans;
            }
        } else{
            conds[r]->initType = initType_W0_cov;
            if( r < (rep*7/8) ){
                conds[r]->initType |= initType_whiteNoise;
            } else {
                conds[r]->initType |= initType_kMeans;
            }
        }
        
        results[r] = new mgResult(nDim, lDim, kDim);
        results[r]->iterCnt = r;
        temps[r] = new mgTemp(nDim, lDim, kDim);
        
//        cout << "0:" << flush;
        
        bayesClusterMG_core(conds[r], data, results[r], temps[r]);
        
//        cout << "1:" << flush;
        
        results[r]->outputToFile(in_name, conds[r], data, r);
        scores.p[r] = results[r]->lowerBounds->p[results[r]->lowerBounds->d1-1];
        scores.outputFineToFile(out_name);
        
//        cout << "2:" << flush;
        
        delete conds[r];
        delete results[r];
        delete temps[r];
    }
    
//    cout << "3:" << flush;
    
    scores.outputFineToFile(out_name);
    
//    cout << "4:" << flush;
    
    free(conds);
    delete data;
    free(results);
    free(temps);
    
    freeRan();
    
    cout << "FINISH: " << (int)(time((time_t *)NULL) - startTime) << " sec. spent." << endl;
    
    return 0;
}

//
