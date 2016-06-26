//
//  bayesClusteringPCA.cpp
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "bayesClusteringPCA.h"
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, const char * argv[]){

    char  in_name[255], out_name[255], baseGamma_name[255];
    time_t  startTime = time((time_t *)NULL);

    int kDim = 0, mDim = 0;
    int maxIteration = 0, rep;
    double lbPosTh = 0.0;
    double maxTFlucAmp, minTFlucAmp;

    // simulated annealing
    double annealSteps = 0;
    //

#ifdef _OPENMP
    cout << "OpenMP defined." << endl;
#endif

    /*******************************
     *  Get command line arguments  *
     *******************************/
    if( (argc != 8) && (argc != 9) ){
        // No file name entered in the command line
        cout << "\nVariational-Bayes Clustering Principal Component Analysis" << endl;
        cout << "    built on 2015.02.10" << endl;
        cout << "Syntax : bcPCA kDim mDim maxIteration lbPosTh annealSteps repetition filename [baseGamma]" << endl;
        cout << "         kDim         : k-dimension, maximum number of cluster" << endl;
        cout << "         mDim         : m-dimension, maximum number of principal components" << endl;
        cout << "                        for each cluster" << endl;
        cout << "         maxIteration : maximum iteration if inference does not reach threshold" << endl;
        cout << "         lbPosTh      : stop iteration if inference reaches this value" << endl;
        cout << "         annealSteps  : iteration for simulated annealing" << endl;
        cout << "         repetition   : number of repetition with an initial condition" << endl;
        cout << "         filename     : name of data file" << endl;
        cout << "         baseGamma    : filename for Gamma distribution for initialization" << endl;
        cout << "Example: bcPCA 5 5 1000 1e-10 500 100 data gamma" << endl << endl;
        exit(1);
    } else if( (argc == 8) || (argc == 9) ){
        // Set parameters
        kDim = atoi(argv[1]);
        mDim = atoi(argv[2]);
        maxIteration = atoi(argv[3]);
        lbPosTh = atof(argv[4]);
        annealSteps = atoi(argv[5]);
        rep = atoi(argv[6]);

        maxTFlucAmp = 1.0 / (double)kDim;
        minTFlucAmp = 1.0e-3;

        // Set the output filename
        strncpy( in_name, argv[7], sizeof(in_name) );

        strncpy( out_name, in_name, sizeof(out_name) );
        strncat( out_name, ".lowerBounds", sizeof(out_name) - strlen(out_name) - 1 );

        if( argc == 9 ){
            strncpy( baseGamma_name, argv[8], sizeof(baseGamma_name) );
        } else {
            memset( baseGamma_name, 0, sizeof(baseGamma_name) );
        }

        cout << " kDim: " << kDim << endl;
        cout << " mDim: " << mDim << endl;
        cout << " maxIteration: " << maxIteration << endl;
        cout << " lbPosTh: " << lbPosTh << endl;
        cout << " annealSteps: " << annealSteps << endl;
        if( annealSteps > 0 ){
            cout << " maxTFlucAmp: " << maxTFlucAmp << endl;
            cout << " minTFlucAmp: " << minTFlucAmp << endl;
        }
        cout << " in_name: " << in_name << endl;
        cout << " out_name: " << out_name << endl;
        if( argc == 9 ){
            cout << " out_name: " << baseGamma_name << endl;
        }
    }
    /*******************************
     *          Print Title         *
     *******************************/
    initRan();
    cout << "*****************************************************************" << endl;
    cout << "*   Variational-Bayes Clustering Principal Component Analysis   *" << endl;
    cout << "*                             by                                *" << endl;
    cout << "*                        OKAMOTO Kenji                          *" << endl;
    cout << "*                            RIKEN                              *" << endl;
    cout << "*****************************************************************" << endl;
    cout << "v 0.01-alpha - built on 2015.02.10" << endl << endl;

#ifdef _OPENMP
    cout << "The number of processors is " << omp_get_num_procs() << endl << endl;
#endif

    int minIteration = 0;
//    double lbNegTh = -1.0e-9;

    int r;

    pcaData *data = new pcaData();
    data->load(in_name, baseGamma_name);
    int nDim = data->nDim, lDim = data->lDim;

    pcaCond **conds = (pcaCond**)malloc( rep * sizeof(pcaCond*) );
    pcaResult **results = (pcaResult**)malloc( rep * sizeof(pcaResult*) );
    pcaTemp **temps = (pcaTemp**)malloc( rep * sizeof(pcaTemp*) );
    Vec1D scores(rep);
    scores.allSet(NAN);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for( r = 0 ; r < rep ; r++ ){

        conds[r] = new pcaCond(kDim, mDim);
        conds[r]->minIteration = minIteration;
        conds[r]->maxIteration = maxIteration;
        conds[r]->lbPosTh = lbPosTh;
        conds[r]->lbNegTh = -0.1 * lbPosTh;

        // thermal fluctuation
        if( annealSteps > 0 ){
            conds[r]->lbNegTh = -0.01 * lbPosTh;
            conds[r]->minIteration = annealSteps;
            conds[r]->annealSteps = annealSteps;

            // -'15.6.29
//            if( data->baseGamma == NULL ){
//                conds[r]->maxTFlucAmp = 1.0 / (double)kDim;  // 0.25;
//            } else {
//                conds[r]->maxTFlucAmp = 0.1;
//            }
//            conds[r]->minTFlucAmp = 1.0e-3;
            //

            conds[r]->maxTFlucAmp = maxTFlucAmp;
            conds[r]->minTFlucAmp = minTFlucAmp;
        }
        //

        if( r < (rep/4) ){
            conds[r]->initType = initType_whiteNoise;
        } else if( r < (rep/2) ){
            conds[r]->initType = initType_randomCluster;
        } else{
            conds[r]->initType = initType_kMeans;
        }
        
        results[r] = new pcaResult(nDim, lDim, kDim, mDim);
        results[r]->iterCnt = r;
        temps[r] = new pcaTemp(nDim, lDim, kDim, mDim);

//        cout << "0:" << flush;

        bayesClusterPCA_core(conds[r], data, results[r], temps[r]);

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
