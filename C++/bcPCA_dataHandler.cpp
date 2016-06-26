//
//  bcPCA_dataHandler.cpp
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.02.05.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "dataUtils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>

int pcaData::load(string filename, string baseGammaFN){

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

        int i, l, n;
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

        setSpectrumBundle(d);  // L x N
        delete d;
    }

    if( baseGammaFN.length() > 0 ){
        fs.open(baseGammaFN.c_str(), ios::in);
        if( !fs.is_open() ){
            return EXIT_FAILURE;
        } else {
            string line;
            
            fs >> line;
            int nDim = atoi(line.c_str());
            fs >> line;
            int kDim = atoi(line.c_str());
            baseGamma = new Mat2D(nDim, kDim);
            
            int i, n, k;
            for( i = 0 ; i < (nDim * kDim) ; i++ ){
                n = i / kDim;
                k = i % kDim;
                
                fs >> line;
                if( fs.eof() )  break;
                
                baseGamma->p[n][k] = atof(line.c_str());
            }
            fs.close();
        }
    }

    return 0;
}


void pcaResult::outputToFile(const char *baseFN, const pcaCond *cond, const pcaData *data, int r){
    int kDim = cond->kDim, mDim = cond->mDim, lDim = data->lDim, nDim = data->nDim;

    fstream fs;
    char filename[255];
    int k, l, m, n;

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

    // W
    sprintf(filename, "%s_%04d_w", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            for( m = 0 ; m < mDim ; m++ ){
                fs << k;
                if( (k < (kDim-1)) || (m < (mDim-1)) ){
                    fs << ", ";
                } else {
                    fs << endl;
                }
            }  // m
        }  // k
        for( k = 0 ; k < kDim ; k++ ){
            for( m = 0 ; m < mDim ; m++ ){
                fs << m;
                if( (k < (kDim-1)) || (m < (mDim-1)) ){
                    fs << ", ";
                } else {
                    fs << endl;
                }
            }  // m
        }  // k
        for( l = 0 ; l < lDim ; l++ ){
            for( k = 0 ; k < kDim ; k++ ){
                for( m = 0 ; m < mDim ; m++ ){
                    fs << Ew->p[k][l][m];
                    if( (k < (kDim-1)) || (m < (mDim-1)) ){
                        fs << ", ";
                    } else {
                        fs << endl;
                    }
                }  // m
            }  // k
        }  // l
        fs.close();
    }

    // alpha
    sprintf(filename, "%s_%04d_alpha", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            fs << k;
            if( k < (kDim-1) ){
                fs << ", ";
            } else {
                fs << endl;
            }
        }  // k
        for( m = 0 ; m < mDim ; m++ ){
            for( k = 0 ; k < kDim ; k++ ){
                fs << Ealp->p[k][m];
                if( k < (kDim - 1) ){
                    fs << ", ";
                } else {
                    fs << endl;
                }
            }  // k
        }  // m
        fs.close();
    }

    // lambda
    sprintf(filename, "%s_%04d_lambda", baseFN, r);
    Elmd->outputToFile(filename);

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

    // zeta
    sprintf(filename, "%s_%04d_zeta", baseFN, r);
    fs.open(filename, ios::out);
    if( fs.is_open() ){
        for( k = 0 ; k < kDim ; k++ ){
            for( m = 0 ; m < mDim ; m++ ){
                fs << k;
                if( (k < (kDim-1)) || (m < (mDim-1)) ){
                    fs << ", ";
                } else {
                    fs << endl;
                }
            }  // m
        }  // k
        for( k = 0 ; k < kDim ; k++ ){
            for( m = 0 ; m < mDim ; m++ ){
                fs << m;
                if( (k < (kDim-1)) || (m < (mDim-1)) ){
                    fs << ", ";
                } else {
                    fs << endl;
                }
            }  // m
        }  // k
        for( n = 0 ; n < nDim ; n++ ){
            for( k = 0 ; k < kDim ; k++ ){
                for( m = 0 ; m < mDim ; m++ ){
                    fs << Ez->p[k][n][m];
                    if( (k < (kDim-1)) || (m < (mDim-1)) ){
                        fs << ", ";
                    } else {
                        fs << endl;
                    }
                }  // m
            }  // k
        }  // n
        fs.close();
    }

}


//
