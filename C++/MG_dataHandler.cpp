//
//  MG_dataHandler.cpp
//  bayesClusteringMixedGauss
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "MG_dataUtils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

int mgData::load(string filename){
    
    fstream fs;
    fs.open(filename.c_str(), ios::in);
    if( !fs.is_open() ){
        return EXIT_FAILURE;
    } else {
        string line;
        
        fs >> line;
        int lDim = atoi(line.c_str());
        //        cout << lDim << ", ";
        fs >> line;
        int nDim = atoi(line.c_str());
        //        cout << nDim << "." << endl;
        Mat2D *d = new Mat2D(lDim, nDim);
        
        int i, l, n = 0;
        for( i = 0 ; i < (lDim * nDim) ; i++ ){
            n = i / lDim;
            l = i % lDim;
            
            fs >> line;
            if( fs.eof() )  break;
            
            d->p[l][n] = atof(line.c_str());
            //            cout << d->p[l][n] << ",";
            //            if( l == (lDim-1) ){
            //                cout << endl;
            //            }
        }
        //        cout << endl;
        
        fs.close();
        
        setDataArray(d);  // L x N
        delete d;
    }
    
    return 0;
}


void mgResult::outputToFile(const char *baseFN, const mgCond *cond, const mgData *data, int r){
    int kDim = cond->kDim, lDim = data->lDim, nDim = data->nDim;
    
    fstream fs;
    char filename[255];
    int i, j, k, l, n;
    
    // lower bounds
    sprintf(filename, "%s_%04d_lb", baseFN, r);
    lowerBounds->outputFineToFile(filename);
    
    // pi
    sprintf(filename, "%s_%04d_pi", baseFN, r);
    Epi->outputToFile(filename);
    
    // Nk
    sprintf(filename, "%s_%04d_Nk", baseFN, r);
    Nk->outputToFile(filename);
    
    // mu
    sprintf(filename, "%s_%04d_mu", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            fs << k;
            if( k < (kDim - 1) ){
                fs << ", ";
            } else {
                fs << endl;
            }
        }  // k
        for( l = 0 ; l < lDim ; l++ ){
            for( k = 0 ; k < kDim ; k++ ){
                fs << Emu->p[k][l];
                if( k < (kDim - 1) ){
                    fs << ", ";
                }
            }  // k
            fs << endl;
        }  // l
        fs.close();
    }
    
//    // Lambda
//    sprintf(filename, "%s_%04d_Sk", baseFN, r);
//    fs.open(filename, ios::out);
//    if( fs.is_open() ){
////        for( k = 0 ; k < kDim ; k++ ){
//            for( i = 0 ; i < lDim ; i++ ){
//                for( j = 0 ; j < lDim ; j++ ){
////                    fs << Sk->p[k][i][j];
//                    fs << W0->p[i][j];
//                    if( j < (lDim-1) ){
//                        fs << ", ";
//                    } else {
//                        fs << endl;
//                    }
//                }  // j(L)
//            }  // i(L)
////        }  // k
//        fs.close();
//    }
    
    // Lambda
    sprintf(filename, "%s_%04d_Lambda", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    fs << Elmd->p[k][i][j];
                    if( j < (lDim-1) ){
                        fs << ", ";
                    } else {
                        fs << endl;
                    }
                }  // j(L)
            }  // i(L)
        }  // k
        fs.close();
    }

    // gamma
    sprintf(filename, "%s_%04d_gamma", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            fs << k;
            if( k < (kDim-1) ){
                fs << ", ";
            }
        }  // k
        fs << endl;
        for( n = 0 ; n < nDim ; n++ ){
            for( k = 0 ; k < kDim ; k++ ){
                fs << gamma->p[n][k];
                if( k < (kDim - 1) ){
                    fs << ", ";
                }
            }  // k
            fs << endl;
        }  // n
        fs.close();
    }
    
}


//
