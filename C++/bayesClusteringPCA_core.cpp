//
//  bayesClusteringPCA_core.cpp
//  bayesClusteringPCA
//
//  Created by OKAMOTO Kenji on 15.01.28.
//  Copyright (c) 2015年 okamoto-kenji. All rights reserved.
//

#include "bayesClusteringPCA_core.h"
#include <Eigen/Dense>

using namespace Eigen;

//#ifdef _OPENMP
//#include <omp.h>
//#endif

#define  MINIMUM_GAMMA_VALUE  1.0e-3
#define  HYPERPARAMETER_VALUE  1.0e-3

int repCount = 0;

void bayesClusterPCA_core(const pcaCond *cond, const pcaData *data, pcaResult *result, pcaTemp *temp){
    
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
        // simulated annealing
        if( itr < cond->annealSteps ){
            result->tFlucAmp = pow( 10.0, log10(cond->maxTFlucAmp) - ((log10(cond->maxTFlucAmp) - log10(cond->minTFlucAmp)) * (double)itr / (double)cond->annealSteps) );
//            result->tFlucAmp = cond->maxTFlucAmp - (cond->maxTFlucAmp - cond->minTFlucAmp) * (double)itr / (double)cond->annealSteps;
//            result->tFlucAmp = fmax( result->tFlucAmp, cond->minTFlucAmp );
        } else {
            result->tFlucAmp = 0.0;
        }
        //
        bayesPcaEStep(cond, data, result, temp);
//        if( (itr % 10) == 0 ){
//            cout << ";" << flush;
//        } else {
//            cout << "," << flush;
//        }

        // M-step
        if( (itr > 0) && (itr == cond->annealSteps) ){
            setParamsFromGamma(cond, data, result, temp, 0.0);
        } else {
            bayesPcaMStep(cond, data, result, temp);
        }
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

}  //  PCAclustering_Core


void bayesPcaEStep(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD;  //, nDimD = d->nDimD;
    int kDim = c->kDim, mDim = c->mDim;
//    double kDimD = c->kDimD, mDimD = c->mDimD;
    
    double maxLnRho, sumRho, sumGam;
    double *rho = t->rho->p, *lnRho = t->lnRho->p;
    MatrixXd matMM(mDim, mDim);   // M x M
    MatrixXd invMatMM(mDim, mDim);   // M x M
    MatrixXd matML(mDim, lDim);             // M x L
    int i, j, k, l, m, n;

    // Z (gamma)
    for( k = 0 ; k < kDim ; k++ ){
        r->Nk->p[k] = 0.0;
    }
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( n = 0 ; n < nDim ; n++ ){
        maxLnRho = -INFINITY;

        for( k = 0 ; k < kDim ; k++ ){
            lnRho[k]  = r->ElnPi->p[k];
            lnRho[k] += lDimD * (r->ElnLmd->p[k] - log(2.0 * M_PI)) / 2.0;
            lnRho[k] -= r->Elmd->p[k] * ElnDistance(c, d, r, t, k, n);

            maxLnRho = fmax( maxLnRho, lnRho[k] );
        }  // k
        sumRho = 0.0;
        for( k = 0 ; k < kDim ; k++ ){
            rho[k] = exp(lnRho[k] - maxLnRho);
            sumRho += rho[k];
        }  // k
        sumGam = 0.0;
        for( k = 0 ; k < kDim ; k++ ){
            // simulated annealing
            if( r->tFlucAmp > 0.0 ){
                r->gamma->p[n][k] = fmax( MINIMUM_GAMMA_VALUE, rho[k] / sumRho + gnoise( r->tFlucAmp ) );
            } else {
                r->gamma->p[n][k] = fmax( MINIMUM_GAMMA_VALUE, rho[k] / sumRho );
            }
            //
            sumGam += r->gamma->p[n][k];
        }  // k
        for( k = 0 ; k < kDim ; k++ ){
            r->gamma->p[n][k] /= sumGam;
            r->Nk->p[k] += r->gamma->p[n][k];
        }  // k
    }  // n

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

    // zeta
//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( k = 0 ; k < kDim ; k++ ){
        for( n = 0 ; n < nDim ; n++ ){
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < mDim ; j++ ){
                    matMM(i,j) = r->gamma->p[n][k] * r->Elmd->p[k] * r->EwTw->p[k][i][j];
                }  // j(M)
            }  // i(M)
            for( i = 0 ; i < mDim ; i++ ){
                matMM(i,i) += 1.0;
            }  // i(M)
            invMatMM = matMM.inverse();
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < mDim ; j++ ){
                    r->sigZeta->p[k][n][i][j] = invMatMM(i,j);
                }  // j(M)
            }  // i(M)

            matML = MatrixXd::Zero(mDim,lDim);
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < lDim ; j++ ){
                    for( m = 0 ; m < mDim ; m++ ){
                        matML(i,j) += r->sigZeta->p[k][n][i][m] * r->Ew->p[k][j][m];
                    }  // m
                    matML(i,j) *= r->gamma->p[n][k] * r->Elmd->p[k];
                }  // j(L)
            }  // i(M)

            for( m = 0 ; m < mDim ; m++ ){
                r->mZeta->p[k][n][m] = 0.0;
                for( l = 0 ; l < lDim ; l++ ){
                        r->mZeta->p[k][n][m] += matML(m,l) * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
                }  // l
            }  // m

            // E[z^2]
            r->Ez2->p[k][n] = 0.0;
            for( i = 0 ; i < mDim ; i++ ){
                // E[z]
                r->Ez->p[k][n][i] = r->mZeta->p[k][n][i];

                // E[zzT]
                for( j = 0 ; j < mDim ; j++ ){
                    r->EzzT->p[k][n][i][j] = r->sigZeta->p[k][n][i][j] + r->mZeta->p[k][n][i] * r->mZeta->p[k][n][j];
                }  // j(M)

                // E[z^2]
                r->Ez2->p[k][n] += r->EzzT->p[k][n][i][i];
            }  // i(M)

        }  // n
    }  // k

}  //  function bayesPcaEStep


double ElnDistance(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t, int k, int n){
    int lDim = d->lDim;
    int mDim = c->mDim;

    int i, j, l, m;
    double term;

    double val = d->xn2->p[n] + r->Emu2->p[k];
    for( i = 0 ; i < mDim ; i++ ){
        for( j = 0 ; j < mDim ; j++ ){
            val += r->EwTw->p[k][i][j] * r->EzzT->p[k][n][j][i];
        }  // j(M)
    }  // i(M)
    val /= 2.0;

    for( l = 0 ; l < lDim ; l++ ){
        term = 0.0;
        for( m = 0 ; m < mDim ; m++ ){
            term += r->Ew->p[k][l][m] * r->Ez->p[k][n][m];
        }  // m
        val -= (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]) * term;

        val -= d->spectrumBundle->p[l][n] * r->Emu->p[k][l];
    }  // l

    return val;
}  // function ElnDistance


void bayesPcaMStep(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD;  //, nDimD = d->nDimD;
    int kDim = c->kDim, mDim = c->mDim;
    //    double kDimD = c->kDimD, mDimD = c->mDimD;
    MatrixXd matMM(mDim, mDim);   // M x M
    MatrixXd invMatMM(mDim, mDim);   // M x M

    int i, j, k, l, m, n;

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( k = 0 ; k < kDim ; k++ ){

    // µ
        r->sigMu_1->p[k] = r->betaMu->p[k] + r->Nk->p[k] * r->Elmd->p[k];
        for( l = 0 ; l < lDim ; l++ ){
            r->mMu->p[k][l] = 0.0;
        }  // l
        for( l = 0 ; l < lDim ; l++ ){
            for( n = 0 ; n < nDim ; n++ ){
                double term = d->spectrumBundle->p[l][n];
                for( m = 0 ; m < mDim ; m++ ){
                    term -= r->Ew->p[k][l][m] * r->Ez->p[k][n][m];
                }  // m
                r->mMu->p[k][l] += term * r->gamma->p[n][k];
            }  // n
            r->mMu->p[k][l] *= r->Elmd->p[k] / r->sigMu_1->p[k];
        }  // l

        // E[mu^2]
        r->Emu2->p[k] = lDimD / r->sigMu_1->p[k];
        for( l = 0 ; l < lDim ; l++ ){
            // E[mu]
            r->Emu->p[k][l] = r->mMu->p[k][l];

            // E[mu^2]
            r->Emu2->p[k] += r->mMu->p[k][l] * r->mMu->p[k][l];
        }  // l

    // W
        matMM = MatrixXd::Zero(mDim,mDim);
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                for( n = 0 ; n < nDim ; n++ ){
                    matMM(i,j) += r->gamma->p[n][k] * r->EzzT->p[k][n][i][j];
                }  // n
                matMM(i,j) *= r->Elmd->p[k];
            }  // j(M)
            matMM(i,i) += r->Ealp->p[k][i];
        }  // i(M)
        invMatMM = matMM.inverse();
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                r->sigW->p[k][i][j] = invMatMM(i,j);
            }  // j(M)
        }  // i(M)

        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                r->mW->p[k][i][j] = 0.0;
            }  // j(M)
        }  // i(L)
        for( l = 0 ; l < lDim ; l++ ){
            for( m = 0 ; m < mDim ; m++ ){
                t->vecM->p[m] = 0.0;
            }  // m
            for( n = 0 ; n < nDim ; n++ ){
                for( m = 0 ; m < mDim ; m++ ){
                    t->vecM->p[m] += r->gamma->p[n][k] * r->Ez->p[k][n][m] * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
                }  // m
            }  // n
            for( m = 0 ; m < mDim ; m++ ){
                for( i = 0 ; i < mDim ; i++ ){
                    r->mW->p[k][l][i] += r->sigW->p[k][i][m] * t->vecM->p[m];
                }  // i(M)
            }  // m
        }  // l
        for( l = 0 ; l < lDim ; l++ ){
            for( m = 0 ; m < mDim ; m++ ){
                r->mW->p[k][l][m] *= r->Elmd->p[k];
            }  // m
        }  // l

    // E[W]
        for( i = 0 ; i < lDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                r->Ew->p[k][i][j] = r->mW->p[k][i][j];
            }  // j(M)
        }  // i(L)

    // E[WTW], E[wm^2]
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                r->EwTw->p[k][i][j] = lDimD * r->sigW->p[k][i][j];
            }  // j(M)
            r->Ewm2->p[k][i] = lDimD * r->sigW->p[k][i][i];
        }  // i(M)
        for( l = 0 ; l < lDim ; l++ ){
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < mDim ; j++ ){
                    r->EwTw->p[k][i][j] += r->mW->p[k][l][i] * r->mW->p[k][l][j];
                }  // j(M)
                r->Ewm2->p[k][i] += r->mW->p[k][l][i] * r->mW->p[k][l][i];
            }  // i(M)
        }  // l

    // alpha
        r->tldAalp->p[k] = r->aAlp->p[k] + lDimD / 2.0;
        for( i = 0 ; i < mDim ; i++ ){
            r->tldBalp->p[k][i] = r->bAlp->p[k] + r->Ewm2->p[k][i] / 2.0;
//        }  // i(M)

    // E[alpha]
//        for( i = 0 ; i < mDim ; i++ ){
            r->Ealp->p[k][i] = r->tldAalp->p[k] / r->tldBalp->p[k][i];
//        }  // i(M)

    // E[ln(alpha)]
//        for( i = 0 ; i < mDim ; i++ ){
            r->ElnAlp->p[k][i] = digamma(r->tldAalp->p[k]) - log(r->tldBalp->p[k][i]);
        }  // i(M)

    // lambda
        r->tldAlmd->p[k] = r->aLmd->p[k] + (r->Nk->p[k] * lDimD) / 2.0;

        r->tldBlmd->p[k] = r->bLmd->p[k];
        for( n = 0 ; n < nDim ; n++ ){
            r->ElnDist->p[k][n] = ElnDistance(c, d, r, t, k, n);
            r->tldBlmd->p[k] += r->gamma->p[n][k] * r->ElnDist->p[k][n];
        }  // n

    // E[lambda], E[ln(lambda)]
        r->Elmd->p[k] = r->tldAlmd->p[k] / r->tldBlmd->p[k];
        r->ElnLmd->p[k] = digamma(r->tldAlmd->p[k]) - log(r->tldBlmd->p[k]);

    }  // k

}  // function bayesPcaMStep


double varLowerBound(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t){
    int nDim = d->nDim;
    double lDimD = d->lDimD, nDimD = d->nDimD;
    int kDim = c->kDim, mDim = c->mDim;
    double kDimD = c->kDimD, mDimD = c->mDimD;

    MatrixXd matMM(mDim, mDim);   // M x M
    MatrixXd invMatMM(mDim, mDim);   // M x M

    int i, j, k, m, n;

    double lnpDQ = r->lnCalp0;
    double lnqZ = 0.0;
    double lnqPi = r->lnChatAlp;
    double lnqZeta = - kDimD * nDimD * mDimD / 2.0;
    double lnqW = - kDimD * lDimD * mDimD / 2.0;
    double lnqAlp = 0.0;
    double lnqMu =  - kDimD * lDimD / 2.0;
    double lnqLmd = 0.0;

//#ifdef _OPENMP
//#pragma omp parallel for
//#endif
    for( k = 0 ; k < kDim ; k++ ){

    // ln(p(D,Theta))
        lnpDQ += (r->Nk->p[k] + r->alp0->p[k] - 1.0) * r->ElnPi->p[k];

        lnpDQ += r->Nk->p[k] * lDimD * (r->ElnLmd->p[k] - log(2.0 * M_PI)) / 2.0;
        for( n = 0 ; n < nDim ; n++ ){
            lnpDQ -= r->gamma->p[n][k] * r->Elmd->p[k] * r->ElnDist->p[k][n];
            lnpDQ -= r->Ez2->p[k][n] / 2.0;

    // ln(q(Z))
            lnqZ += r->gamma->p[n][k] * log(r->gamma->p[n][k]);

    // ln(q(zeta))
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < mDim ; j++ ){
                    matMM(i,j) = r->sigZeta->p[k][n][i][j];
                }  // j(M)
            }  // i(M)
            lnqZeta = - log(matMM.determinant()) / 2.0;
            invMatMM = matMM.inverse();
            for( i = 0 ; i < mDim ; i++ ){
                for( j = 0 ; j < mDim ; j++ ){
                    lnqZeta -= invMatMM(i,j) * r->sigZeta->p[k][n][i][j] / 2.0;
                }  // j
            }  // i
        } // n

    // ln(q(alpha))
        lnqAlp -= mDimD * lngamma(r->tldAalp->p[k]);

        for( m = 0 ; m < mDim ; m++ ){
            lnpDQ += (r->aAlp->p[k] + lDimD/2.0 - 1.0) * r->ElnAlp->p[k][m];
            lnpDQ -= (r->bAlp->p[k] + r->Ewm2->p[k][m]/2.0) * r->Ealp->p[k][m];

        // ln(q(alpha))
            lnqAlp += r->tldAalp->p[k] * log(r->tldBalp->p[k][m]);
            lnqAlp += (r->tldAalp->p[k] - 1.0) * r->ElnAlp->p[k][m];
            lnqAlp -= r->tldBalp->p[k][m] * r->Ealp->p[k][m];

        }  // m
        lnpDQ += - mDimD * lngamma(r->aAlp->p[k]) + mDimD * r->aAlp->p[k] * log(r->bAlp->p[k]);

//        lnpDQ += lDimD * log(r->betaMu->p[k]) / 2.0 - r->betaMu->p[k] * r->Emu2->p[k] / 2.0;
        lnpDQ += (lDimD * log(r->betaMu->p[k]) - r->betaMu->p[k] * r->Emu2->p[k]) / 2.0;

        lnpDQ += - lngamma(r->aLmd->p[k]) + r->aLmd->p[k] * log(r->bLmd->p[k]);
        lnpDQ += (r->aLmd->p[k] - 1.0) * r->ElnLmd->p[k] - r->bLmd->p[k] * r->Elmd->p[k];

    // ln(q(pi))
        lnqPi += (r->alpPi->p[k] - 1.0) * r->ElnPi->p[k];

    // ln(q(W))
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                matMM(i,j) = r->sigW->p[k][i][j];
            }  // j(M)
        }  // i(M)
        lnqW -= lDimD * log(matMM.determinant()) / 2.0;
        invMatMM = matMM.inverse();
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                lnqW -= lDimD * invMatMM(i,j) * r->sigW->p[k][i][j] / 2.0;
            }  // j(M)
        }  // i(M)

    // ln(q(mu))
        lnqMu += lDimD * log(r->sigMu_1->p[k]) / 2.0;

    // ln(q(lambda))
        lnqLmd += - lngamma(r->tldAlmd->p[k]) + r->tldAlmd->p[k] * log(r->tldBlmd->p[k]);
        lnqLmd += (r->tldAlmd->p[k] - 1.0) * r->ElnLmd->p[k] - r->tldBlmd->p[k] * r->Elmd->p[k];

    }  // k

    return  lnpDQ - lnqZ - lnqPi - lnqZeta - lnqW - lnqAlp - lnqMu - lnqLmd;
}  // function varLowerBound


void setInitial(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD, nDimD = d->nDimD;
    int kDim = c->kDim, mDim = c->mDim;
    double kDimD = c->kDimD;  //, mDimD = c->mDimD;
    
    int i, j, k, l, m, n;

    double sm, sd;  //, ip;		// sum, standard deviation, inner product
    double term;
    double W_fluc = 0.5;

    // hyperparameters
    r->lnCalp0 = 0.0;
    term = 0.0;
    for( k = 0 ; k < kDim ; k++ ){
        r->alp0->p[k] = 1.0;
        term += r->alp0->p[k];
        r->lnCalp0 -= lngamma(r->alp0->p[k]);

        r->betaMu->p[k] = HYPERPARAMETER_VALUE;
        r->aAlp->p[k]   = HYPERPARAMETER_VALUE;
        r->bAlp->p[k]   = HYPERPARAMETER_VALUE;
        r->aLmd->p[k]   = HYPERPARAMETER_VALUE;
        r->bLmd->p[k]   = HYPERPARAMETER_VALUE;
    }  // k
    r->lnCalp0 += lngamma(term);

    if( d->baseGamma != NULL ){                                                                         // base Gamma
#pragma omp critical
        cout << "init(" << repCount++ << ":" << r->iterCnt << "): base Gamma" << endl;
        double sumG;
        for( n = 0 ; n < nDim ; n++ ){
            sumG = 0.0;
            for( k = 0 ; k < kDim ; k++ ){
//                r->gamma->p[n][k] = fmax( d->baseGamma->p[n][k] + gnoise(0.1), MINIMUM_GAMMA_VALUE);
                r->gamma->p[n][k] = fmax( d->baseGamma->p[n][k] + gnoise(0.05), MINIMUM_GAMMA_VALUE);
                sumG += r->gamma->p[n][k];
            }
            for( k = 0 ; k < kDim ; k++ ){
                r->gamma->p[n][k] /= sumG;
            }  // k
        }
        W_fluc = 0.0;  // 0.1;

    } else if( (c->initType == initType_noSpecification) || (c->initType == initType_whiteNoise) ){                  // white noise
#pragma omp critical
        cout << "init(" << repCount++ << ":" << r->iterCnt << "): white noise" << endl;
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

    } else if( (c->initType == initType_randomCluster) || (c->initType == initType_kMeans) ) {                            // random cluster / k-means
        int *index = (int*)malloc( nDim * sizeof(int) );
        for( n = 0 ; n < nDim ; n++ ){
            index[n] = randomInteger(0, kDim-1);
        }  // n
        double ck = 0.0, dis, minDis;

        if( c->initType == initType_kMeans ){  // K-means switch
#pragma omp critical
            cout << "init(" << repCount++ << ":" << r->iterCnt << "): k-means" << endl;

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
                                kmAvg.p[k][l] += d->spectrumBundle->p[l][n];
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
                            dis += pow(d->spectrumBundle->p[l][n] - kmAvg.p[k][l], 2.0);
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
            cout << "init(" << repCount++ << ":" << r->iterCnt << "): random cluster" << endl;
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
        cout << "init(" << repCount++ << ":" << r->iterCnt << "): unknown init." << endl;
    }

    for( k = 0 ; k < kDim ; k++ ){
        r->Nk->p[k] = 0.0;
    }  // k
    for( n = 0 ; n < nDim ; n++ ){
        for( k = 0 ; k < kDim ; k++ ){
            r->Nk->p[k] += r->gamma->p[n][k];
        }  // k
    }  // n

    setParamsFromGamma(c, d, r, t, W_fluc);

//    // π
//    for( k = 0 ; k < kDim ; k++ ){
//        r->Epi->p[k] = 0.0;
//        for( n = 0 ; n < nDim ; n++ ){
//            r->Epi->p[k] += r->gamma->p[n][k];
//        }  // n
//        r->Epi->p[k] /= nDimD;
//        r->ElnPi->p[k] = log(r->Epi->p[k]);
//    }  // k
//
//    // μ
//    for( k = 0 ; k < kDim ; k++ ){
//        r->Emu2->p[k] = 0.0;
//        for( l = 0 ; l < lDim ; l++ ){
//            r->Emu->p[k][l] = 0.0;
//            for( n = 0 ; n < nDim ; n++ ){
//                r->Emu->p[k][l] += r->gamma->p[n][k] * d->spectrumBundle->p[l][n];
//            }  // n
//            r->Emu->p[k][l] /= r->Nk->p[k];
//            r->Emu2->p[k] += r->Emu->p[k][l] * r->Emu->p[k][l];
//        }  // l
//    }  // k
//
//    // W, lambda
//    double fluc = 1.0;  // fluctuation
//    double sig2, nume, deno;
//    for( k = 0 ; k < kDim ; k++ ){
//
//        Mat2D tempDat(lDim, nDim), pcaRes(lDim, mDim);
//        for( l = 0 ; l < lDim ; l++ ){
//            for( n = 0 ; n < nDim ; n++ ){
//                tempDat.p[l][n] = r->gamma->p[n][k] * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
//            }  // n
//        }  // l
//        pca( &tempDat, &pcaRes );
//
//        for( m = 0 ; m < mDim ; m++ ){		// add fluctuation
//            sm = 0.0;
//            for( l = 0 ; l < lDim ; l++ ){
//                r->Ew->p[k][l][m] = pcaRes.p[l][m];
//                sm += pcaRes.p[l][m] * pcaRes.p[l][m];
//            }  // l
//            sd = sqrt(sm / lDimD);
//            for( l = 0 ; l < lDim ; l++ ){
//                r->Ew->p[k][l][m] += gnoise(fluc * sd);
//            }  // l
//        }  // m
//        for( m = 0 ; m < mDim ; m++ ){		// normalize to unit vectors
//            sm = 0.0;
//            for( l = 0 ; l < lDim ; l++ ){
//                sm += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
//            }  // l
//            sm = sqrt(sm);
//            for( l = 0 ; l < lDim ; l++ ){
//                r->Ew->p[k][l][m] /= sm;
//            }  // l
//        }  // m
//
//        // zeta
//        for( m = 0 ; m < mDim ; m++ ){
//            sm = 0.0;
//            for( n = 0 ; n < nDim ; n++ ){
//                deno = 0.0;
//                nume = 0.0;
//                for( l = 0 ; l < lDim ; l++ ){
//                    nume += r->Ew->p[k][l][m] * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
//                    deno += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
//                }  // l
//                r->Ez->p[k][n][m] = nume / deno;
//
//                sm += r->Ez->p[k][n][m] * r->Ez->p[k][n][m];
//            }  // n
//            sd = sqrt( sm / nDimD );
//            for( n = 0 ; n < nDim ; n++ ){
//                r->Ez->p[k][n][m] /= sd;
//            }  // n
//            for( l = 0 ; l < lDim ; l++ ){
//                r->Ew->p[k][l][m] *= sd;
//            }  // l
//        }  // m
//
//
//        // W-derivatives
//        for( i = 0 ; i < mDim ; i++ ){
//            for( j = 0 ; j < mDim ; j++ ){
//                r->EwTw->p[k][i][j] = 0.0;
//                for( l = 0 ; l < lDim ; l++ ){
//                    r->EwTw->p[k][i][j] += r->Ew->p[k][l][i] * r->Ew->p[k][l][j];
//                }  // l
//            }  // j(M)
//        }  // i(M)
//        for( m = 0 ; m < mDim ; m++ ){
//            r->Ewm2->p[k][m] = 0.0;
//            for( l = 0 ; l < lDim ; l++ ){
//                r->Ewm2->p[k][m] += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
//            }  // l
//        }  // m
//
//        // alpha
//        for( m = 0 ; m < mDim ; m++ ){
//            r->Ealp->p[k][m] = 0.0;
//            term = 0.0;
//            for( l = 0 ; l < lDim ; l++ ){
//                term += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
//            }  // l
//            r->Ealp->p[k][m] = lDimD / term;
//            r->ElnAlp->p[k][m] = log(r->Ealp->p[k][m]);
//        }  // m
//
//        // zeta-derivatives
//        for( n = 0 ; n < nDim ; n++ ){
//            r->Ez2->p[k][n] = 0.0;
//            for( i = 0 ; i < mDim ; i++ ){
//                r->Ez2->p[k][n] += r->Ez->p[k][n][i] * r->Ez->p[k][n][i];
//                for( j = 0 ; j < mDim ; j++ ){
//                    r->EzzT->p[k][n][i][j] = r->Ez->p[k][n][i] * r->Ez->p[k][n][j];
//                }  // j(M)
//            }  // i(M)
//        }  // n
//
//        // lambda
//        sig2 = 0.0;
//        Vec1D res(lDim);
//        for( n = 0 ; n < nDim ; n++ ){
//            for( l = 0 ; l < lDim ; l++ ){
//                res.p[l] = d->spectrumBundle->p[l][n] - r->Emu->p[k][l];
//            }  // l
//            for( m = 0 ; m < mDim ; m++ ){
//                for( l = 0 ; l < lDim ; l++ ){
//                    res.p[l] -= r->Ez->p[k][n][m] * r->Ew->p[k][l][m];
//                }  // l
//            }  // m
//            sig2 += r->gamma->p[n][k] * res.var();
//        }  // n
//        r->Elmd->p[k] = 1.0 / (sig2 / r->Nk->p[k]);
//        r->ElnLmd->p[k] = log(r->Elmd->p[k]);
//
//    }  // k

}  // function setInitial


void setParamsFromGamma(const pcaCond *c, const pcaData *d, pcaResult *r, pcaTemp *t, double W_fluc){
    int lDim = d->lDim, nDim = d->nDim;
    double lDimD = d->lDimD, nDimD = d->nDimD;
    int kDim = c->kDim, mDim = c->mDim;
    double kDimD = c->kDimD;  //, mDimD = c->mDimD;
    
    int i, j, k, l, m, n;
    
    double sm, sd;  //, ip;		// sum, standard deviation, inner product
    double term;

    // π
    for( k = 0 ; k < kDim ; k++ ){
        r->Epi->p[k] = 0.0;
        for( n = 0 ; n < nDim ; n++ ){
            r->Epi->p[k] += r->gamma->p[n][k];
        }  // n
        r->Epi->p[k] /= nDimD;
        r->ElnPi->p[k] = log(r->Epi->p[k]);
    }  // k
    
    // μ
    for( k = 0 ; k < kDim ; k++ ){
        r->Emu2->p[k] = 0.0;
        for( l = 0 ; l < lDim ; l++ ){
            r->Emu->p[k][l] = 0.0;
            for( n = 0 ; n < nDim ; n++ ){
                r->Emu->p[k][l] += r->gamma->p[n][k] * d->spectrumBundle->p[l][n];
            }  // n
            r->Emu->p[k][l] /= r->Nk->p[k];
            r->Emu2->p[k] += r->Emu->p[k][l] * r->Emu->p[k][l];
        }  // l
    }  // k
    
    // W, lambda
    double fluc = W_fluc;   // 1.0;  // fluctuation
    double sig2, nume, deno;
    for( k = 0 ; k < kDim ; k++ ){
        
        Mat2D tempDat(lDim, nDim), pcaRes(lDim, mDim);
        for( l = 0 ; l < lDim ; l++ ){
            for( n = 0 ; n < nDim ; n++ ){
                tempDat.p[l][n] = r->gamma->p[n][k] * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
            }  // n
        }  // l
        pca( &tempDat, &pcaRes );
        
        for( m = 0 ; m < mDim ; m++ ){		// add fluctuation
            sm = 0.0;
            for( l = 0 ; l < lDim ; l++ ){
                r->Ew->p[k][l][m] = pcaRes.p[l][m];
                sm += pcaRes.p[l][m] * pcaRes.p[l][m];
            }  // l
            sd = sqrt(sm / lDimD);
            for( l = 0 ; l < lDim ; l++ ){
                r->Ew->p[k][l][m] += gnoise(fluc * sd);
            }  // l
        }  // m
        for( m = 0 ; m < mDim ; m++ ){		// normalize to unit vectors
            sm = 0.0;
            for( l = 0 ; l < lDim ; l++ ){
                sm += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
            }  // l
            sm = sqrt(sm);
            for( l = 0 ; l < lDim ; l++ ){
                r->Ew->p[k][l][m] /= sm;
            }  // l
        }  // m
        
        // zeta
        for( m = 0 ; m < mDim ; m++ ){
            sm = 0.0;
            for( n = 0 ; n < nDim ; n++ ){
                deno = 0.0;
                nume = 0.0;
                for( l = 0 ; l < lDim ; l++ ){
                    nume += r->Ew->p[k][l][m] * (d->spectrumBundle->p[l][n] - r->Emu->p[k][l]);
                    deno += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
                }  // l
                r->Ez->p[k][n][m] = nume / deno;
                
                sm += r->Ez->p[k][n][m] * r->Ez->p[k][n][m];
            }  // n
            sd = sqrt( sm / nDimD );
            for( n = 0 ; n < nDim ; n++ ){
                r->Ez->p[k][n][m] /= sd;
            }  // n
            for( l = 0 ; l < lDim ; l++ ){
                r->Ew->p[k][l][m] *= sd;
            }  // l
        }  // m
        
        
        // W-derivatives
        for( i = 0 ; i < mDim ; i++ ){
            for( j = 0 ; j < mDim ; j++ ){
                r->EwTw->p[k][i][j] = 0.0;
                for( l = 0 ; l < lDim ; l++ ){
                    r->EwTw->p[k][i][j] += r->Ew->p[k][l][i] * r->Ew->p[k][l][j];
                }  // l
            }  // j(M)
        }  // i(M)
        for( m = 0 ; m < mDim ; m++ ){
            r->Ewm2->p[k][m] = 0.0;
            for( l = 0 ; l < lDim ; l++ ){
                r->Ewm2->p[k][m] += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
            }  // l
        }  // m
        
        // alpha
        for( m = 0 ; m < mDim ; m++ ){
            r->Ealp->p[k][m] = 0.0;
            term = 0.0;
            for( l = 0 ; l < lDim ; l++ ){
                term += r->Ew->p[k][l][m] * r->Ew->p[k][l][m];
            }  // l
            r->Ealp->p[k][m] = lDimD / term;
            r->ElnAlp->p[k][m] = log(r->Ealp->p[k][m]);
        }  // m
        
        // zeta-derivatives
        for( n = 0 ; n < nDim ; n++ ){
            r->Ez2->p[k][n] = 0.0;
            for( i = 0 ; i < mDim ; i++ ){
                r->Ez2->p[k][n] += r->Ez->p[k][n][i] * r->Ez->p[k][n][i];
                for( j = 0 ; j < mDim ; j++ ){
                    r->EzzT->p[k][n][i][j] = r->Ez->p[k][n][i] * r->Ez->p[k][n][j];
                }  // j(M)
            }  // i(M)
        }  // n
        
        // lambda
        sig2 = 0.0;
        Vec1D res(lDim);
        for( n = 0 ; n < nDim ; n++ ){
            for( l = 0 ; l < lDim ; l++ ){
                res.p[l] = d->spectrumBundle->p[l][n] - r->Emu->p[k][l];
            }  // l
            for( m = 0 ; m < mDim ; m++ ){
                for( l = 0 ; l < lDim ; l++ ){
                    res.p[l] -= r->Ez->p[k][n][m] * r->Ew->p[k][l][m];
                }  // l
            }  // m
            sig2 += r->gamma->p[n][k] * res.var();
        }  // n
        r->Elmd->p[k] = 1.0 / (sig2 / r->Nk->p[k]);
        r->ElnLmd->p[k] = log(r->Elmd->p[k]);

    }  // k

}


//
