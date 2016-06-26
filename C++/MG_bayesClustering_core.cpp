//
//  MG_bayesClustering_core.cpp
//  bayesClusteringMixedGauss
//
//  Created by OKAMOTO Kenji on 15.03.04.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "MG_bayesClustering_core.h"
#include <Eigen/Dense>

using namespace Eigen;

//#ifdef _OPENMP
//#include <omp.h>
//#endif

#define  MINIMUM_GAMMA_VALUE  1.0e-3
#define  HYPERPARAMETER_VALUE  1.0e-3

int repCount = 0;

void bayesClusterMG_core(const mgCond *cond, const mgData *data, mgResult *result, mgTemp *temp){
    
    int minIter = cond->minIteration;
    int maxIter = cond->maxIteration;
    double lbPosTh = cond->lbPosTh;
    double lbNegTh = cond->lbNegTh;
    
    // init
    setInitial(cond, data, result, temp);
//    result->outputToFile("init", cond, data, result->iterCnt);
    // end of init
    
    int itr, lbLen;
    double lb, lbDiff;
    Vec1D *lbVec = result->lowerBounds;     // lower-bound series
    
    for( itr = 0 ; itr < maxIter ; itr++ ){
        
        // E-step
        bayesMG_EStep(cond, data, result, temp);
//        if( (itr % 10) == 0 ){
//            cout << ";" << flush;
//        } else {
//            cout << "," << flush;
//        }
        
        // M-step
        bayesMG_MStep(cond, data, result, temp);
//        if( (itr % 10) == 0 ){
//            cout << ":" << flush;
//        } else {
//            cout << "." << flush;
//        }
        
        
        //////////  judge
        // ln likelihood
        lbLen = (int)lbVec->d1;
        lbVec->resize(lbLen + 1);
//        cout << "+" << flush;
        
        lb = varLowerBound(cond, data, result, temp);
        lbVec->p[lbLen] = lb;
        
        if( lbLen > 0 ){
            lbDiff = (lbVec->p[lbLen] - lbVec->p[lbLen-1]) / fabs(lbVec->p[lbLen]);
            if( (lbDiff < lbPosTh) && (lbDiff > lbNegTh) && (itr >= minIter) ){
//            if( (lbDiff < lbPosTh) && (itr >= minIter) ){
                break;
            }
        }
        //////////  judge
//        cout << "-" << flush;
        
    }
//    cout << "done." << endl;
    
}  //  clusteringMG_Core


void bayesMG_EStep(const mgCond *c, const mgData *d, mgResult *r, mgTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD;  //, nDimD = d->nDimD;
    int kDim = c->kDim;
//    double kDimD = c->kDimD;
    
    double maxLnRho, sumRho, sumGam;
    double *rho = t->rho->p, *lnRho = t->lnRho->p;
    int i, j, k, l, n;
    
    // Z (gamma)
    for( k = 0 ; k < kDim ; k++ ){
        r->Nk->p[k] = 0.0;
        for( l = 0 ; l < lDim ; l++ ){
            r->Exk->p[k][l] = 0.0;
        }  // l
    }  // k
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( n = 0 ; n < nDim ; n++ ){
        maxLnRho = -INFINITY;
        
        for( k = 0 ; k < kDim ; k++ ){
            lnRho[k]  = r->ElnPi->p[k];
            lnRho[k] += (r->ElnLmd->p[k] - lDimD * log(2.0 * M_PI)) / 2.0;
            lnRho[k] -= r->ElnDist->p[k][n] / 2.0;

            maxLnRho = fmax( maxLnRho, lnRho[k] );
        }  // k
        sumRho = 0.0;
        for( k = 0 ; k < kDim ; k++ ){
            rho[k] = exp(lnRho[k] - maxLnRho);
            sumRho += rho[k];
        }  // k
        sumGam = 0.0;
        for( k = 0 ; k < kDim ; k++ ){
            r->gamma->p[n][k] = fmax( MINIMUM_GAMMA_VALUE, rho[k] / sumRho );
            sumGam += r->gamma->p[n][k];
        }  // k
        for( k = 0 ; k < kDim ; k++ ){
            r->gamma->p[n][k] /= sumGam;

            r->Nk->p[k] += r->gamma->p[n][k];
            for( l = 0 ; l < lDim ; l++ ){
                r->Exk->p[k][l] += r->gamma->p[n][k] * d->dataArray->p[l][n];
            }  // l
        }  // k
    }  // n
//    cout << r->gamma->p[0][0] << ",";
    for( k = 0 ; k < kDim ; k++ ){
        for( l = 0 ; l < lDim ; l++ ){
            r->Exk->p[k][l] /= r->Nk->p[k];
        }  // l
    }  // k
    for( k = 0 ; k < kDim ; k++ ){
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                r->Sk->p[k][i][j] = 0.0;
                for( n = 0 ; n < nDim ; n++ ){
                    r->Sk->p[k][i][j] += r->gamma->p[n][k] * (d->dataArray->p[i][n] - r->Exk->p[k][i]) * (d->dataArray->p[j][n] - r->Exk->p[k][j]);
                }  // n
                r->Sk->p[k][i][j] /= r->Nk->p[k];
            }  // j(L)
        }  // i(L)
    }  // k
    
    // π
    r->hatAlpPi = 0.0;
    for( k = 0 ; k < kDim ; k++ ){
        r->alpPi->p[k] = r->alp0->p[k] + r->Nk->p[k];
        r->hatAlpPi += r->alpPi->p[k];
    }  // k
    r->lnChatAlp = lngamma(r->hatAlpPi);
    for( k = 0 ; k < kDim ; k++ ){
        r->Epi->p[k] = r->alpPi->p[k] / r->hatAlpPi;
        r->ElnPi->p[k] = digamma(r->alpPi->p[k]) - digamma(r->hatAlpPi);
        r->lnChatAlp -= lngamma(r->alpPi->p[k]);
    }  // k
    
}  //  function bayesMG_EStep


void bayesMG_MStep(const mgCond *c, const mgData *d, mgResult *r, mgTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD;  //, nDimD = d->nDimD;
    int kDim = c->kDim;
//    double kDimD = c->kDimD;
    MatrixXd matLL(lDim, lDim);   // L x L
    MatrixXd invMatLL(lDim, lDim);   // L x L
    
    int i, j, k, l, n;
    double term;
    
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( k = 0 ; k < kDim ; k++ ){
        
        // µ, Lambda

        // beta_k
        r->beta_k->p[k] = r->beta0 + r->Nk->p[k];

        // m_k
        for( l = 0 ; l < lDim ; l++ ){
            r->mk->p[k][l] = (r->beta0 * r->m0->p[l] + r->Nk->p[k] * r->Exk->p[k][l]) / r->beta_k->p[k];
        }  // l

        // W_k
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                matLL(i,j) = r->invW0->p[i][j];
            }  // j(L)
        }  // i(L)
        term = (r->beta0 * r->Nk->p[k]) / (r->beta0 + r->Nk->p[k]);
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                matLL(i,j) += r->Nk->p[k] * r->Sk->p[k][i][j];
                matLL(i,j) += term * (r->Exk->p[k][i] - r->m0->p[i]) * (r->Exk->p[k][j] - r->m0->p[j]);
            }  // j(L)
        }  // i(L)
        invMatLL = matLL.inverse();
        r->maxWk->p[k] = -INFINITY;
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                r->Wk->p[k][i][j] = invMatLL(i,j);
                r->maxWk->p[k] = max( r->maxWk->p[k], fabs(r->Wk->p[k][i][j]) );
            }  // j(L)
        }  // i(L)

        // nu_k
        r->nu_k->p[k] = r->nu0 + r->Nk->p[k];


        // E[mu]
        for( l = 0 ; l < lDim ; l++ ){
            r->Emu->p[k][l] = r->mk->p[k][l];
        }  // l

        // E[Lamnda]
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                r->Elmd->p[k][i][j] = r->nu_k->p[k] * r->Wk->p[k][i][j];
            }  // j(L)
        }  // i(L)
        
        // E[ln(Lmd)]
        r->ElnLmd->p[k] = lDimD * log(2.0);
        for( i = 0 ; i < lDim ; i++ ){
            r->ElnLmd->p[k] += digamma((r->nu_k->p[k] + 1.0 - (double)i) / 2.0);

            for( j = 0 ; j < lDim ; j++ ){
                matLL(i,j) = r->Wk->p[k][i][j] / r->maxWk->p[k];
            }  // j(L)
        }  // i(L)
        r->ElnLmd->p[k] += log( matLL.determinant() ) + lDimD * log( r->maxWk->p[k] );

        // E[(x-mu)^T Lmd (x-mu)]
        for( n = 0 ; n < nDim ; n++ ){
            r->ElnDist->p[k][n] = lDimD / r->beta_k->p[k];
            for( i = 0 ; i < lDim ; i++ ){
                double term = 0.0;
                for( j = 0 ; j < lDim ; j++ ){
                    term += r->Wk->p[k][i][j] * (d->dataArray->p[j][n] - r->mk->p[k][j]);
                }  // j(L)
                r->ElnDist->p[k][n] += r->nu_k->p[k] * (d->dataArray->p[i][n] - r->mk->p[k][i]) * term;
            }  // i(L)
        }  // n

    }  // k
    
}  // function bayesMG_MStep


double varLowerBound(const mgCond *c, const mgData *d, mgResult *r, mgTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD;  //, nDimD = d->nDimD;
    int kDim = c->kDim;
    double kDimD = c->kDimD;
    
    MatrixXd matLL(lDim, lDim);   // L x L
    MatrixXd invMatLL(lDim, lDim);   // L x L
    
    int i, j, k, n;
    
    double lnpDQ = 0.0;
    double lnpZ = 0.0;
    double lnpPi = r->lnCalp0;
    double lnpMuLmd = kDimD * log( Wishart_lnB(r->W0, r->nu0, r->maxW0) );

    double lnqZ = 0.0;
    double lnqPi = r->lnChatAlp;
    double lnqMuLmd = 0.0;

    double term, term2;
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( k = 0 ; k < kDim ; k++ ){
        
        // ln(p(D,Theta))
        term = r->ElnLmd->p[k] - lDimD * (1.0/r->beta_k->p[k] + log(2.0 * M_PI)) ;
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                term -= r->nu_k->p[k] * r->Sk->p[k][i][j] * r->Wk->p[k][j][i];
            }  // j(L)
        }  // i(L)
        for( i = 0 ; i < lDim ; i++ ){
            term2 = 0.0;
            for( j = 0 ; j < lDim ; j++ ){
                term2 += r->Wk->p[k][i][j] * (r->Exk->p[k][j] - r->mk->p[k][j]);
            }  // j(L)
            term -= r->nu_k->p[k] * (r->Exk->p[k][i] - r->mk->p[k][i]) * term2;
        }  // i(L)
        lnpDQ += r->Nk->p[k] * term / 2.0;
        
        for( n = 0 ; n < nDim ; n++ ){
            // ln(p(Z))
            lnpZ += r->gamma->p[n][k] * r->ElnPi->p[k];

            // ln(q(Z))
            lnpZ += r->gamma->p[n][k] * log(r->gamma->p[n][k]);
        } // n

        // ln(p(pi))
        lnpPi += (r->alp0->p[k] - 1.0) * r->ElnPi->p[k];
        
        // ln(p(mu,Lambda))
        term  = lDimD * log(r->beta0 / 2.0 / M_PI);
        term += r->ElnLmd->p[k] - lDimD * r->beta0 / r->beta_k->p[k];
        for( i = 0 ; i < lDim ; i++ ){
            term2 = 0.0;
            for( j = 0 ; j < lDim ; j++ ){
                term2 += r->Wk->p[k][i][j] * (r->mk->p[k][j] - r->m0->p[j]);
            }  // j(L)
            term -= r->beta0 * r->nu_k->p[k] * (r->mk->p[k][i] - r->m0->p[i]) * term2;
        }  // i(L)
        lnpMuLmd += term / 2.0;
        lnpMuLmd += (r->nu0 - lDimD - 1.0) * r->ElnLmd->p[k] / 2.0;
        for( i = 0 ; i < lDim ; i++ ){
            term = 0.0;
            for( j = 0 ; j < lDim ; j++ ){
                term += r->invW0->p[i][j] * r->Wk->p[k][j][i];
            }  // j(L)
            lnpMuLmd -= r->nu_k->p[k] * term / 2.0;
        }  // i(L)

        // ln(q(pi))
        lnqPi += (r->alpPi->p[k] - 1.0) * r->ElnPi->p[k];

        // ln(q(mu,Lambda))
        lnqMuLmd += r->ElnLmd->p[k] / 2.0;
        lnqMuLmd += lDimD * (log(r->beta_k->p[k] / 2.0 / M_PI) - 1.0) / 2.0;
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                t->matLL->p[i][j] = r->Wk->p[k][i][j];
            }  // j(L)
        }  // i(L)
        lnqMuLmd -= Wishart_H(t->matLL, r->nu_k->p[k], r->ElnLmd->p[k], r->maxWk->p[k]);

    }  // k
    
    return  lnpDQ + lnpZ + lnpPi + lnpMuLmd - lnqZ - lnqPi - lnqMuLmd;
}  // function varLowerBound


void setInitial(const mgCond *c, const mgData *d, mgResult *r, mgTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD, nDimD = d->nDimD;
    int kDim = c->kDim;
    double kDimD = c->kDimD;
    
    int i, j, k, l, n;
    double sm;		// sum
    double term;

    int initType_gamma = c->initType & initType_gamma_mask;

#pragma omp critical
    cout << "init(" << repCount++ << ":" << r->iterCnt << "): ";

    // hyperparameters
    r->lnCalp0 = 0.0;
    term = 0.0;
    for( k = 0 ; k < kDim ; k++ ){
        r->alp0->p[k] = 1.0;
        term += r->alp0->p[k];
        r->lnCalp0 -= lngamma(r->alp0->p[k]);
        
//        r->beta_k->p[k] = HYPERPARAMETER_VALUE;
    }  // k
    r->lnCalp0 += lngamma(term);

    r->beta0 = HYPERPARAMETER_VALUE;    // ???
    r->nu0 = lDimD;      // ???
    for( l = 0 ; l < lDim ; l++ ){
        r->m0->p[l] = 0.0;      // ???
    }  // l

    switch( c->initType & initType_W0_mask ){
        case initType_W0_one:
        default:{
#pragma omp critical
            cout << "W0-one - ";
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    r->W0->p[i][j] = (i == j) ? (1.0 / r->nu0) : 0.0;     // ???
                }  // j(L)
            }  // i(L)
            r->maxW0 = 1.0 / r->nu0;
        }
            break;

        case initType_W0_hyp:{
#pragma omp critical
            cout << "W0-hyp - ";
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    r->W0->p[i][j] = (i == j) ? HYPERPARAMETER_VALUE : 0.0;     // ???
                }  // j(L)
            }  // i(L)
            r->maxW0 = HYPERPARAMETER_VALUE;
        }
            break;

        case initType_W0_cov:{
#pragma omp critical
            cout << "W0-cov - ";
            Vec1D mean(lDim);
            for( l = 0 ; l < lDim ; l++ ){
                mean.p[l] = 0.0;
            }  // l
            for( l = 0 ; l < lDim ; l++ ){
                for( n = 0 ; n < nDim ; n++ ){
                    mean.p[l] += d->dataArray->p[l][n];
                }  // n
                mean.p[l] /= nDimD;
            }  // l
            MatrixXd cov = MatrixXd::Zero(lDim,lDim), invCov(lDim,lDim);
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    for( n = 0 ; n < nDim ; n++ ){
                        cov(i,j) += (d->dataArray->p[i][n] - mean.p[i]) * (d->dataArray->p[j][n] - mean.p[j]);
                    }  // n
                    cov(i,j) /= nDimD;
                }  // j(L)
            }  // i(L)
            invCov = cov.inverse();
            r->maxW0 = -INFINITY;
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    r->W0->p[i][j] = invCov(i,j) / r->nu0;
                    r->maxW0 = max( r->maxW0, fabs(r->W0->p[i][j]) );
                }  // j(L)
            }  // i(L)
        }
            break;

        case initType_W0_var:{
#pragma omp critical
            cout << "W0-var - ";
            Vec1D mean(lDim);
            for( l = 0 ; l < lDim ; l++ ){
                mean.p[l] = 0.0;
            }  // l
            for( l = 0 ; l < lDim ; l++ ){
                for( n = 0 ; n < nDim ; n++ ){
                    mean.p[l] += d->dataArray->p[l][n];
                }  // n
                mean.p[l] /= nDimD;
            }  // l
            Vec1D var(lDim);
            for( l = 0 ; l < lDim ; l++ ){
                for( n = 0 ; n < nDim ; n++ ){
                    var.p[l] += (d->dataArray->p[l][n] - mean.p[l]) * (d->dataArray->p[l][n] - mean.p[l]);
                }  // n
                var.p[l] /= nDimD;
            }  // l
            r->maxW0 = -INFINITY;
            for( i = 0 ; i < lDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    r->W0->p[i][j] = 0.0;
                }  // j(L)
                r->W0->p[i][i] = 1.0 / var.p[i] / r->nu0;
                r->maxW0 = max( r->maxW0, fabs(r->W0->p[i][i]) );
            }  // i(L)
        }
            break;

    }
    MatrixXd matLL(lDim,lDim), invMatLL(lDim,lDim);
    for( i = 0 ; i < lDim ; i++ ){
        for( j = 0 ; j < lDim ; j++ ){
            matLL(i,j) = r->W0->p[i][j];
        }  // j(L)
    }  // i(L)
    invMatLL = matLL.inverse();
    for( i = 0 ; i < lDim ; i++ ){
        for( j = 0 ; j < lDim ; j++ ){
            r->invW0->p[i][j] = invMatLL(i,j);
        }  // j(L)
    }  // i(L)

    if( (initType_gamma == initType_noSpecification) || (initType_gamma == initType_whiteNoise) ){                  // white noise
#pragma omp critical
        cout << "white noise" << endl;
        double sumG;
        for( n = 0 ; n < nDim ; n++ ){
            sumG = 0.0;
            for( k = 0 ; k < kDim ; k++ ){
                r->gamma->p[n][k] = 3.0 + enoise(2.0);
                sumG += r->gamma->p[n][k];
            }  // k
            for( k = 0 ; k < kDim ; k++ ){
                r->gamma->p[n][k] /= sumG;
            }  // k
        }  // n
        
    } else if( (initType_gamma == initType_randomCluster) || (initType_gamma == initType_kMeans) ) {                            // random cluster / k-means
        int *index = (int*)malloc( nDim * sizeof(int) );
        for( n = 0 ; n < nDim ; n++ ){
            index[n] = randomInteger(0, kDim-1);
        }  // n
        double ck = 0.0, dis, minDis;
        
        if( initType_gamma == initType_kMeans ){  // K-means switch
#pragma omp critical
            cout << "k-means" << endl;
            
            int *lastIndex = (int*)malloc( nDim * sizeof(int) );
            for( n = 0 ; n < nDim ; n++ ){
                lastIndex[n] = index[n];
            }  // n
            
            Mat2D kmAvg(kDim, lDim);
            do{                          // calc avg.
                
                kmAvg.allSet(0.0);
                for( k = 0 ; k < kDim ; k++ ){
//                    sprintf wvnm, "kmAvg%dWv", k;
//                    make /O/N=(lDim) $wvnm;
//                    wave wv = $wvnm;
//                    wv = 0;
                    ck = 0.0;
                    for( n = 0 ; n < nDim ; n += 1 ){
                        if( index[n] == k ){
                            for( l = 0 ; l < lDim ; l += 1 ){
                                kmAvg.p[k][l] += d->dataArray->p[l][n];
                            }  //l
                            ck += 1.0;
                        }
                    }  //n
                    for( l = 0 ; l < lDim ; l += 1 ){
                        kmAvg.p[k][l] /= ck;
                    }  //l
                }  //k
                
                // assign index
                for( n = 0 ; n < nDim ; n++ ){
                    minDis = INFINITY;
                    for( k = 0 ; k < kDim ; k++ ){
                        dis = 0.0;
                        for( l = 0 ; l < lDim ; l++ ){
                            dis += pow(d->dataArray->p[l][n] - kmAvg.p[k][l], 2.0);
                        }  //l
                        dis = sqrt(dis);
                        if( dis < minDis ){
                            index[n] = k;
                            minDis = dis;
                        }
                    }  //k
                }  //n
                
                // check
                for( n = 0 ; n < nDim ; n++ ){
                    if( index[n] != lastIndex[n] ){
                        break;
                    }
                }  //n
                if( n == nDim ){
                    break;
                } else {
                    for( n = 0 ; n < nDim ; n++ ){
                        lastIndex[n] = index[n];
                    }  // n
                }
            }while(1);
            free(lastIndex);
        } else {  // K-means
            
#pragma omp critical
            cout << "random cluster" << endl;
        }
        
        int initK, lastK;
        switch( 1 ){
            case 0:
            default:
                for( n = 0 ; n < nDim ; n++ ){
                    initK = index[n];
                    r->gamma->p[n][initK] = 3.0/(kDimD + 2.0) + enoise(0.02);
                    for( k = 1 ; k < kDim-1 ; k++ ){
                        r->gamma->p[n][initK+k % kDim] = max( 0.01, 1.0/(kDimD + 2.0) + enoise(0.02) );
                    }  // k
                    lastK = (initK + kDim - 1) % kDim;
                    r->gamma->p[n][lastK] = 1.0;
                    for( k = 0 ; k < (kDim-1) ; k += 1 ){
                        r->gamma->p[n][lastK] -= r->gamma->p[n][(initK + k) % kDim];
                    }  // k
                }  // n
                break;
                
            case 1:
                for( n = 0 ; n < nDim ; n++ ){
                    for( k = 0 ; k < kDim ; k++ ){
                        r->gamma->p[n][k] = 0.0;
                    }  // k
                    r->gamma->p[n][index[n]] += 0.5;
                    sm = 0.0;
                    for( k = 0 ; k < kDim ; k++ ){
                        r->gamma->p[n][k] += 1 + enoise(0.1);
                        sm += r->gamma->p[n][k];
                    }  // k
                    for( k = 0 ; k < kDim ; k++ ){
                        r->gamma->p[n][k] /= sm;
                    }  // k
                }
                break;
        }
        
        free(index);
        
    } else {
#pragma omp critical
        cout << "unknown init." << endl;
    }
    
    for( k = 0 ; k < kDim ; k++ ){
        
        r->Nk->p[k] = 0.0;
        for( n = 0 ; n < nDim ; n++ ){
            r->Nk->p[k] += r->gamma->p[n][k];
        }  // n

        for( l = 0 ; l < lDim ; l++ ){
            r->Exk->p[k][l] = 0.0;
            for( n = 0 ; n < nDim ; n++ ){
                r->Exk->p[k][l] += r->gamma->p[n][k] * d->dataArray->p[l][n];
            }  // n
            r->Exk->p[k][l] /= r->Nk->p[k];
            r->Emu->p[k][l] = r->Exk->p[k][l];
        }  // l

        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < lDim ; j++ ){
                r->Sk->p[k][i][j] = 0.0;
                for( n = 0 ; n < nDim ; n++ ){
                    r->Sk->p[k][i][j] += r->gamma->p[n][k] * (d->dataArray->p[i][n] - r->Exk->p[k][i]) * (d->dataArray->p[j][n] - r->Exk->p[k][j]);
                }  // n
                r->Sk->p[k][i][j] /= r->Nk->p[k];
            }  // j(L)
        }  // i(L)
    }  // k


    // π
    for( k = 0 ; k < kDim ; k++ ){
        r->Epi->p[k] = 0.0;
        for( n = 0 ; n < nDim ; n++ ){
            r->Epi->p[k] += r->gamma->p[n][k];
        }  // n
        r->Epi->p[k] /= nDimD;
        r->ElnPi->p[k] = log(r->Epi->p[k]);
    }  // k

    bayesMG_MStep(c, d, r, t);
//    MatrixXd tempMat(lDim,lDim), invMat(lDim,lDim);;
//    // Lambda
//    for( k = 0 ; k < kDim ; k++ ){
//        
//        for( i = 0 ; i < lDim ; i++ ){
//            for( j = 0 ; j < lDim ; j++ ){
//                tempMat(i,j) = r->Sk->p[k][i][j];
//            }  // j(L)
//        }  // i(L)
//        invMat = tempMat.inverse();
//        for( i = 0 ; i < lDim ; i++ ){
//            for( j = 0 ; j < lDim ; j++ ){
//                r->Elmd->p[k][i][j] = invMat(i,j);
//            }  // j(L)
//        }  // i(L)
//        r->ElnLmd->p[k] = log(invMat.determinant());
//
//        for( n = 0 ; n < nDim ; n++ ){
//            r->ElnDist->p[k][n] = 0.0;
//            for( i = 0 ; i < lDim ; i++ ){
//                term = 0.0;
//                for( j = 0 ; j < lDim ; j++ ){
//                    term += r->Elmd->p[k][i][j] * (d->dataArray->p[j][n] - r->Emu->p[k][j]);
//                }  // j(L)
//                r->ElnDist->p[k][n] += (d->dataArray->p[i][n] - r->Emu->p[k][i]) * term;
//            }  // i(L)
//        }  // n
//        
//    }  // k
    
}  // function setInitial

//
