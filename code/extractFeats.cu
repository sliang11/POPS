#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "Eigen/Dense"
using namespace Eigen;

#define INF 2e9
#define MAX_FEAT_DIMS 10
#define MAX_INDICATORS 20
 
extern __shared__ float shMem[]; 


template <class T> __host__ __device__ void swap(T &x, T &y){
	T tmp = x;
	x = y;
	y = tmp;
}

__host__ __device__ int nextInt(double val, bool eq){
    int intVal;
    if (val == ceil(val))
        intVal = eq ? val : val + 1;
    else
        intVal = ceil(val);
    return intVal;
}

__host__ __device__ int lastInt(double val, bool eq) {
    int intVal;
    if (val == floor(val))
        intVal = eq ? val : val - 1;
    else
        intVal = floor(val);
    return intVal;
}

__host__ __device__ double power(double val, int exponent) {
    double ret = 1;
    for (int i = 0; i < exponent; i++) {
        ret *= val;
    }
    return ret;
}

__device__ inline int nvToNUnit(int nv, int unitNv, int ptsPerStep) {
    return nv > unitNv ? (nv - unitNv) / ptsPerStep + 1 : 0; 
}

__device__ inline int getNSubCurScale(int exNv, int featNv, int featPtsPerStep) {
    return 1 + lastInt((double)(exNv - featNv) / featPtsPerStep, 1);
}


__device__ int getDeviceZInd(int ix, int iy, int ny, int nExY, int exNx, int exNy, int nExOnDevice, int nExProcessed) {


    int ixEx, nExLastCol, nExTotal = nExOnDevice + nExProcessed;
    long long zInd = (long long) ix * ny + iy;
    ixEx = ix / exNx;

    if (ixEx == ceil((float)nExTotal / nExY) - 1) {
        nExLastCol = nExTotal - nExY * ixEx; 
        zInd -= (ix - ixEx * exNx) * (ny - exNy * nExLastCol);
    }

    if (ixEx == ceil((float)nExProcessed / nExY) - 1) { 
        zInd -= ixEx * nExY * exNx * exNy;
        nExLastCol = nExProcessed - nExY * ixEx;
        zInd -= (ix + 1 - ixEx * exNx) * exNy * nExLastCol;

    } else {
        zInd -= nExProcessed * exNx * exNy;
    }

    return zInd;

}

// go-right move
__device__ void slideAlongX (float* z, int offsetShMem, int initOffsetShMem, int maxFloatInShMem, int exId, int nExProcessed, 
    int exNx, int exNy, int prevShNx, int curShNx, int shNy, int nExY, int ny, 
    int prevLx, int prevLy, int xStep, int nExOnDevice, int useIncreShMem) {

    int i, ix, iy, ind;
    if (useIncreShMem) {
        int numIncreX = xStep + curShNx - prevShNx;
        int shMemIndBase = offsetShMem - initOffsetShMem + (curShNx - numIncreX) * shNy;
        for (i = 0; i < ceil((float)numIncreX * shNy / blockDim.x); i++) {
            ind = i * blockDim.x + threadIdx.x; 
            if (ind < numIncreX * shNy) { 

                ix = (exId / nExY) * exNx + (prevLx + xStep + curShNx - numIncreX) + ind / shNy;
                iy = (exId % nExY) * exNy + prevLy + ind % shNy; 

                shMem[initOffsetShMem + (shMemIndBase + ind) % (maxFloatInShMem - initOffsetShMem)] =
                    z[getDeviceZInd(ix, iy, ny, nExY, exNx, exNy, nExOnDevice, nExProcessed)];
            }
        }
    } else {
        for (i = 0; i < ceil((float)curShNx * shNy / blockDim.x); i++) {
            ind = i * blockDim.x + threadIdx.x;
            if (ind < curShNx * shNy) {
                ix = (exId / nExY) * exNx + (prevLx + xStep) + ind / shNy;
                iy = (exId % nExY) * exNy + prevLy + ind % shNy;

                shMem[initOffsetShMem + ind] = z[getDeviceZInd(ix, iy, ny, nExY, exNx, exNy, nExOnDevice, nExProcessed)];
            }

        }
    }
}

__device__ int getMaxI(int li, int curShNv, int featPtsPerStep, int exNv, int featNv) {
    int maxI = li + curShNv - featNv;
    maxI = lastInt((float)maxI / featPtsPerStep, 1) * featPtsPerStep;
    return maxI;
}

__device__ void getMinMaxIAndNextMinI(int& minI, int& maxI, int& nextMinI, const int numFeatScales, 
    int li, int curShNv, int featPtsPerStep, int exNv, int featNv, int iFeatScale, bool isX) {

    minI = shMem[numFeatScales * (isX ? 5 : 6) + iFeatScale];
    maxI = getMaxI(li, curShNv, featPtsPerStep, exNv, featNv);
    nextMinI = minI > maxI ? minI : maxI + featPtsPerStep;
}

__device__ void updateMinIInShMem(int& minI, int& maxI, int& nextMinI, const int numFeatScales, 
    int li, int curShNv, int featPtsPerStep, int exNv, int featNv, int iFeatScale, bool isX) {
    
    getMinMaxIAndNextMinI(minI, maxI, nextMinI, numFeatScales, li, curShNv, featPtsPerStep, exNv, featNv, iFeatScale, isX);
    shMem[numFeatScales * (isX ? 5 : 6) + iFeatScale] = nextMinI;
}

__device__ inline double getXorY(int ixOrIyInSub, int subIxOrIy, double resol) {
    return (subIxOrIy + ixOrIyInSub) * resol;
}

__device__ double getZ(int ixInSub, int iyInSub, int prevLx, int prevLy, int subIx, int subIy, int offsetShMem, int initOffsetShMem, int shNy, int maxFloatInShMem) {
    int shIx, shIy;
    shIx = subIx + ixInSub - prevLx; shIy = subIy + iyInSub - prevLy;
    return shMem[initOffsetShMem + (offsetShMem - initOffsetShMem + (shIx * shNy + shIy)) % (maxFloatInShMem - initOffsetShMem)];
}

__device__ inline double var (double sum, double sum2, int num) {
    return sum2 / num - (sum * sum) / (num * num) > 0 ? sum2 / num - (sum * sum) / (num * num) : 0;
}

__device__ inline double stdv (double sum, double sum2, int num) {
    return sqrtf(var(sum, sum2, num));
}

__device__ inline double skewness (double avg, double stdev, double sum, double sum2, double sum3, int num) {
    return stdev == 0 ? 0 : (sum3 - 3 * avg * sum2 + 3 * power(avg, 2) * sum - num * power(avg, 3)) / (num * power(stdev, 3));
}

__device__ inline double kurtosis (double avg, double stdev, double sum, double sum2, double sum3, double sum4, int num) {
    return stdev == 0 ? -3 : (sum4 - 4 * avg * sum3 + 6 * power(avg, 2) * sum2 - 4 * power(avg, 3) * sum + num * power(avg, 4)) / (num * power(stdev, 4)) - 3;
}

__device__ inline double ED(double x1, double x2, double y1, double y2, double z1 = INF, double z2 = INF) {
    return sqrtf((z1 == INF || z2 == INF) ? power(x1 - x2, 2) + power(y1 - y2, 2) : power(x1 - x2, 2) + power(y1 - y2, 2) + power(z1 - z2, 2));
}

__device__ inline double covAfterZNorm(double cov, double sumA, double sumB, double meanA, double meanB, double stdA, double stdB, int n) {
    return (cov - sumA * meanB) / (stdA * stdB); 

__device__ void core_PCA_ISPRS12_gridded (double* featsPerSub, int prevLx, int prevLy, int subIx, int subIy, int shNy, int featNx, int featNy, int offsetShMem, int initOffsetShMem, int maxFloatInShMem, 
    double resolX, double resolY, bool stdNormXY, bool stdNormZ, 
    double distTh = INF, int numValidTh = 3) {

    double x, y, z, xc, yc, zc, tmpX, tmpY, tmpZ, meanX, meanY, meanZ, stdX, stdY, stdZ, covMat[9], sumEig; 
    int i, ixInSub, iyInSub, numValid;
    tmpX = tmpY = tmpZ = covMat[0] = covMat[1] = covMat[2] = covMat[4] = covMat[5] = covMat[8] = 0;

    ixInSub = lastInt((double)featNx / 2, 0); iyInSub = lastInt((double)featNy / 2, 0);
    xc = getXorY(ixInSub, subIx, resolX); 
    yc = getXorY(iyInSub, subIy, resolY);
    zc = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);

    numValid = 0;
    for (ixInSub = 0; ixInSub < featNx; ixInSub++) {
        x = getXorY(ixInSub, subIx, resolX);
        for (iyInSub = 0; iyInSub < featNy; iyInSub++) {
            y = getXorY(iyInSub, subIy, resolY);
            z = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);

            if (ED(xc, x, yc, y, zc, z) < distTh) { 
                tmpX += x; tmpY += y; tmpZ += z;
                covMat[0] += x * x; covMat[4] += y * y; covMat[8] += z * z; 
                covMat[1] += x * y; covMat[2] += x * z; covMat[5] += y * z;
                numValid++;
            }
        }
    }

    if (numValid >= numValidTh) {
        meanX = tmpX / numValid; meanY = tmpY / numValid; meanZ = tmpZ / numValid;
        if (stdNormXY) {
            stdX = stdv(tmpX, covMat[0], numValid); stdY = stdv(tmpY, covMat[4], numValid);
        } else {
            stdX = stdY = 1;
        }
        if (stdNormZ) {
            stdZ = stdv(tmpZ, covMat[8], numValid);
        } else {
            stdZ = 1;
        }
        
        covMat[0] = covAfterZNorm(covMat[0], tmpX, tmpX, meanX, meanX, stdX, stdX, numValid);
        covMat[4] = covAfterZNorm(covMat[4], tmpY, tmpY, meanY, meanY, stdY, stdY, numValid);
        covMat[8] = covAfterZNorm(covMat[8], tmpZ, tmpZ, meanZ, meanZ, stdZ, stdZ, numValid);
        covMat[1] = covAfterZNorm(covMat[1], tmpX, tmpY, meanX, meanY, stdX, stdY, numValid);
        covMat[2] = covAfterZNorm(covMat[2], tmpX, tmpZ, meanX, meanZ, stdX, stdZ, numValid);
        covMat[5] = covAfterZNorm(covMat[5], tmpY, tmpZ, meanY, meanZ, stdY, stdZ, numValid);
        covMat[3] = covMat[1]; covMat[6] = covMat[2]; covMat[7] = covMat[5]; //对称阵
        for (i = 0; i < 9; i++)
            covMat[i] /= numValid - 1;

        SelfAdjointEigenSolver<Matrix3d> eig(3);
        eig.computeDirect(Map<Matrix3d>(covMat), EigenvaluesOnly);
        for (i = 0; i < 2; i++) {
            featsPerSub[i] = eig.eigenvalues()[2 - i];
        }
        sumEig = eig.eigenvalues()[0] + eig.eigenvalues()[1] + eig.eigenvalues()[2];
        for (i = 0; i < 2; i++) {
            featsPerSub[i] /= sumEig;
        }

        tmpX = (1 - (featsPerSub[1] - featsPerSub[0])) / 2; 
        tmpY = 1 - tmpX; 

        tmpZ = 3 * eig.eigenvalues()[0] / sumEig;
        tmpX = (1 - tmpZ) * ED(0.5, tmpX, 0.5, tmpY) / ED(0.5, 1, 0.5, 0);
        tmpY = 1 - tmpX - tmpZ;
        featsPerSub[0] = max(0.5 * (2 * tmpY + tmpZ), 0.0);
        featsPerSub[1] = max(0.5 * sqrtf(3) * tmpZ, 0.0);

    } else {
        featsPerSub[0] = featsPerSub[1] = INF;
    }

    
}

__device__ bool checkValidity(int ixInSub, int iyInSub, int featNy, double resolX, double resolY, int subIx, int subIy, int prevLx, int prevLy, 
    int offsetShMem, int initOffsetShMem, int shNy, int maxFloatInShMem, 
    unsigned int* indiValid, double& z, const double xc, const double yc, const double zc, const double distTh) {
    
    double x, y;
    const int i = ixInSub * featNy + iyInSub;

    if (i < 32 * MAX_INDICATORS) {
        if (((indiValid[i / 32] >> (i % 32)) & 1) == 0) {
            return false;
        } 
        z = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);
    } else {
        x = getXorY(ixInSub, subIx, resolX);
        y = getXorY(iyInSub, subIy, resolY);
        z = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);
        if (ED(xc, x, yc, y, zc, z) >= distTh) {
            return false;
        }
    }
    return true;
}

__device__ void getRanks(int& rank, int& rankAmongCands, double val, int& nRepeat, int numValid, double l_filter, double r_filter,
    int startIInSub, int finishIInSub, double resolX, double resolY, 
    int featNy, unsigned int* indiValid, 
    const double xc, const double yc, const double zc, 
    int subIx, int subIy, int prevLx, int prevLy, 
    int offsetShMem, int initOffsetShMem, int shNy, int maxFloatInShMem, const double distTh) {

    double z;
    int cnt, i, ixInSub, iyInSub;
    bool isValid, isCand, metSelf, met_lf, met_rf;
    rank = rankAmongCands = cnt = metSelf = met_lf = met_rf = nRepeat = 0;
    for (i = startIInSub; i <= finishIInSub; i++) {

        ixInSub = i / featNy; iyInSub = i % featNy;
        isValid = checkValidity(ixInSub, iyInSub, featNy, resolX, resolY, subIx, subIy, prevLx, prevLy, 
                offsetShMem, initOffsetShMem, shNy, maxFloatInShMem, indiValid, z, xc, yc, zc, distTh);
        if (!isValid) {
            continue;
        }
        
        if (z > l_filter && z < r_filter) {
            isCand = true;
        } else if (z < l_filter || z > r_filter) {
            isCand = false;
        } else {
            if (z == l_filter) {
                if (met_lf) { 
                    isCand = true;
                } else {
                    isCand = false;
                    met_lf = true;
                }
            }
            if (z == r_filter) { 
                if (met_rf) {
                    isCand = true;
                } else {
                    isCand = false;
                    met_rf = true;
                }
            }
        }

        if (z < val) {
            rank++;
            if (isCand) {
                rankAmongCands++;
            }
        } else if (z == val) { 
            if (!metSelf) {
                rank++;
                rankAmongCands++; 
                metSelf = 1;
            }
            nRepeat++;
        }

        cnt++;
        if (cnt == numValid) {
            break;
        }
    } 
}

__device__ double selection_basic_algorithm(int k, int r, double l_filter, double r_filter, double randProp,
    int n, int startIInSub, int finishIInSub, double resolX, double resolY, 
    int featNy, unsigned int* indiValid, 
    const double xc, const double yc, const double zc, 
    int subIx, int subIy, int prevLx, int prevLy, 
    int offsetShMem, int initOffsetShMem, int shNy, int maxFloatInShMem, const double distTh){
    
    int i, iu_r, ixInSub, iyInSub, cnt, cnt_lf, cnt_rf, repeats, rank, rankAmongCands;
    double x, y, z, u;
    bool isValid, isCand;


    int i_round = -1;
    while (true) {

        i_round++;
        if (l_filter == r_filter) {
            return l_filter;
        }

        // Step 1: find the pivot u
        iu_r = randProp * r; 
        cnt = cnt_lf = cnt_rf = 0;
        u = INF;
        for (i = startIInSub; i <= finishIInSub; i++) {
            ixInSub = i / featNy; iyInSub = i % featNy;
            isValid = checkValidity(ixInSub, iyInSub, featNy, resolX, resolY, subIx, subIy, prevLx, prevLy, 
                    offsetShMem, initOffsetShMem, shNy, maxFloatInShMem, indiValid, z, xc, yc, zc, distTh);
            if (!isValid) {
                continue;
            }

            if (z > l_filter && z < r_filter) {
                isCand = true;
            } else if (z < l_filter || z > r_filter) {
                isCand = false;
            } else {
                if (z == l_filter) {
                    isCand = cnt_lf; 
                    cnt_lf++;
                } else if (z == r_filter) {
                    isCand = cnt_rf;
                    cnt_rf++;
                }
            }

            if (isCand) {
                if (z != l_filter && z != r_filter) {
                    u = z; 
                }
                if (cnt >= iu_r && u != INF) { 
                    break;
                }
                cnt++;
            }
        }

        if (u == INF) { 
            return cnt_lf >= k ? l_filter : r_filter;
        }

        //Step 2: compare all valid points / candiates with u to determine u's rank / rankAmongCands. 注意rank与左右filter无关，即它也考虑<= l_filter和>= r_filter的点
        getRanks(rank, rankAmongCands, u, repeats, n, l_filter, r_filter, startIInSub, finishIInSub, resolX, resolY, featNy, 
            indiValid, xc, yc, zc, subIx, subIy, prevLx, prevLy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem, distTh);

        //Step 3: adjust filters and go to next round
        if (k >= rank && k <= rank + repeats - 1) {
            return u;
        }   
        if (k < rank) { 
            r_filter = u;
            r = rankAmongCands - 1; 
        } else {
            l_filter = u;
            r -= rankAmongCands - 1 + repeats; 
        }

    }

}


__device__ void core_stats_gridded (double* featsPerSub, int prevLx, int prevLy, int subIx, int subIy, int shNy, int featNx, int featNy, int offsetShMem, int initOffsetShMem, int maxFloatInShMem, 
    double resolX, double resolY, bool stdNormZ, float randProp, double distTh = INF, int numValidTh = 3) { 

    double x, y, z, z_, meanZ, stdZ, sums[4], gamma; 
    int i, j, u, ixInSub, iyInSub, numValid, numValid_, rank, k[3];
    bool metSelf;
    unsigned int indiValid[MAX_INDICATORS];  
    
    for (i = 0; i < MAX_INDICATORS; i++) {
        indiValid[i] = 0;
    }
    for (i = 0; i < 4; i++) {
        sums[i] = 0;
    }

    ixInSub = lastInt((double)featNx / 2, 0); iyInSub = lastInt((double)featNy / 2, 0); 
    const double xc = getXorY(ixInSub, subIx, resolX); 
    const double yc = getXorY(iyInSub, subIy, resolY);
    const double zc = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);

    numValid = 0;
    featsPerSub[3] = INF; featsPerSub[7] = -INF;
    for (ixInSub = 0; ixInSub < featNx; ixInSub++) {
        x = getXorY(ixInSub, subIx, resolX);
        for (iyInSub = 0; iyInSub < featNy; iyInSub++) {

            y = getXorY(iyInSub, subIy, resolY);
            z = getZ(ixInSub, iyInSub, prevLx, prevLy, subIx, subIy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem);

            if (ED(xc, x, yc, y, zc, z) < distTh) {
                for (i = 1; i <= 4; i++) {
                    sums[i - 1] += power(z, i);
                }

                if (z < featsPerSub[3]) {
                    featsPerSub[3] = z;
                } 
                if (z > featsPerSub[7]) {
                    featsPerSub[7] = z;
                }

                i = ixInSub * featNy + iyInSub;
                if (i < 32 * MAX_INDICATORS) { 
                    j = i % 32; i /= 32;
                    indiValid[i] |= (1 << j);
                }
                numValid++;

            }
        }
    }

    if (numValid >= numValidTh) {
        
        // prepare for normalization
        meanZ = sums[0] / numValid;
        stdZ = stdv(sums[0], sums[1], numValid);

        //q1, median, q3 (before normalization)
        for (i = 0; i < 3; i++) {
            k[i] = floor(1.0 / 3 + (numValid + 1.0 / 3) * (i + 1) / 4.0);

            if (k[i] < 1 || k[i] + 1 > numValid) {
                if (k[i] < 1) {
                    featsPerSub[4 + i] = featsPerSub[3];
                }
                if (k[i] + 1 > numValid) {
                    featsPerSub[4 + i] = featsPerSub[7];
                }
            } else {

                z = selection_basic_algorithm(k[i], numValid - 2, featsPerSub[3], featsPerSub[7], randProp, numValid, 0, featNx * featNy - 1, 
                    resolX, resolY, featNy, indiValid, xc, yc, zc, subIx, subIy, prevLx, prevLy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem, 
                    distTh);
                
                z_ = selection_basic_algorithm(k[i] + 1, numValid - 2, featsPerSub[3], featsPerSub[7], randProp, numValid, 0, featNx * featNy - 1, 
                    resolX, resolY, featNy, indiValid, xc, yc, zc, subIx, subIy, prevLx, prevLy, offsetShMem, initOffsetShMem, shNy, maxFloatInShMem, 
                    distTh);

                if (z == z_) {
                    featsPerSub[4 + i] = z;
                } else {
                    gamma = numValid * (i + 1) / 4.0 - k[i] + ((i + 1) / 4.0 + 1) / 3;
                    featsPerSub[4 + i] = (1 - gamma) * z + gamma * z_;
                }                
            }
        }

        //normalized min, q1, median, q3, max
        if (stdZ != 0) {
            for (i = 3; i < 8; i++) {
                featsPerSub[i] = (featsPerSub[i] - meanZ) / (stdNormZ ? stdZ : 1);
            }
            featsPerSub[0] = stdNormZ ? 1 : stdZ;
            featsPerSub[1] = skewness(meanZ, stdZ, sums[0], sums[1], sums[2], numValid);  //skewness和kurtosis具有平移和横向拉伸不变性
            featsPerSub[2] = kurtosis(meanZ, stdZ, sums[0], sums[1], sums[2], sums[3], numValid);
        } else {
            for (i = 0; i < 8; i++) {
                featsPerSub[i] = i == 2 ? -3 : 0;
            }
        }
        
    } else {
        for (i = 0; i < 8; i++) {
            featsPerSub[i] = INF;
        }
    }
}

extern "C" __global__
void extractFeats_gridded (float* feats, float* z, float* sortedFeatScales, int* featNxByScale, int* featNyByScale, 
    int* featXPtsPerStepByScale, int* featYPtsPerStepByScale, float* randProps,
    int numFeatScales,int ny, 
    float resolX_f, float resolY_f, int exNx, int exNy, 
    int maxFloatInShMem, int isVolumetric, int stdNormXY, int stdNormZ, int nExProcessed, 
    int coreId, int featDims, int numValidTh = 3, int shNx_ = INF, int shNy_ = INF, 
    int useIncreShmem = 1, int useMultiScale = 1) { 

    int i, j, m, n, incre, offsetShMem, cumNumSub, prevLx, prevLy, xStep, yStep, numFeatScalesX, numFeatScalesY, curNumFeatScales,
        nExY, minIx, maxIx, minIy, maxIy, nextMinIx, nextMinIy, subIx, subIy, shNx, shNy, prevShNx, curShNx, featNx, featNy, nSubX, nSubY, nSub, nSubTotal, curNSubY, curNSub,
        featXPtsPerStep, featYPtsPerStep, exId;
    double scale, distTh, featsPerSub[MAX_FEAT_DIMS], resolX = resolX_f, resolY = resolY_f;

    if (threadIdx.x < numFeatScales) {
        shMem[threadIdx.x] = sortedFeatScales[threadIdx.x];
        featNx = shMem[numFeatScales + threadIdx.x] = featNxByScale[threadIdx.x];
        featNy = shMem[numFeatScales * 2 + threadIdx.x] = featNyByScale[threadIdx.x];
        featXPtsPerStep = shMem[numFeatScales * 3 + threadIdx.x] = featXPtsPerStepByScale[threadIdx.x];
        featYPtsPerStep = shMem[numFeatScales * 4 + threadIdx.x] = featYPtsPerStepByScale[threadIdx.x];
        
        shMem[numFeatScales * 5 + threadIdx.x] = 0; 
        shMem[numFeatScales * 6 + threadIdx.x] = 0; 
    }

    __syncthreads();

    nSubTotal = 0;
    for (i = 0; i < numFeatScales; i++) {
        nSubTotal += getNSubCurScale(exNx, shMem[numFeatScales + i], shMem[numFeatScales * 3 + i]) * 
            getNSubCurScale(exNy, shMem[numFeatScales * 2 + i], shMem[numFeatScales * 4 + i]);
    }
    const int initOffsetShMem = 7 * numFeatScales;

    n = exNx * exNy;
    m = min(maxFloatInShMem - initOffsetShMem, exNx * exNy);
    scale = sortedFeatScales[numFeatScales - 1];
    if (m == n) {
        shNx = exNx; shNy = exNy;
    } else {
        shNy = min(exNy, (int)sqrtf(m * resolX / resolY) / 16 * 16);
        shNy = max(shNy, (int)ceil(scale / resolY));
        shNx = min(exNx, m / shNy);
        i = min(exNx, (int)sqrtf(m * resolY / resolX) / 16 * 16);
        i = max(i, (int)ceil(scale / resolX));  
        j = min(exNy, m / shNx); 
        
        if (abs(shNy - shNx) > abs(i - j)) { 
            shNx = i; shNy = j;
        }
    }

    shNx = min(shNx, shNx_); shNy = min(shNy, shNy_);
    maxFloatInShMem = shNx * shNy + initOffsetShMem; 

    nExY = nvToNUnit(ny, exNy, exNy); 
    exId = nExProcessed + blockIdx.x;

    //The main loop
    offsetShMem = initOffsetShMem; xStep = prevShNx = curShNx = shNx; prevLx = -shNx; prevLy = 0;
    curNumFeatScales = numFeatScales;
    while(1) {

        //go right
        slideAlongX(z, offsetShMem, initOffsetShMem, maxFloatInShMem, exId, nExProcessed,
            exNx, exNy, prevShNx, curShNx, shNy, nExY, ny, prevLx, prevLy, xStep, gridDim.x, useIncreShmem);    
        prevLx += xStep;
        __syncthreads();

        //run core method
        m = 0; 
        cumNumSub = 0;
        numFeatScalesX = 0; xStep = INF;
        for (i = 0; i < curNumFeatScales; i++) {
            
            scale = shMem[i];
            featNx = shMem[numFeatScales + i]; 
            featNy = shMem[numFeatScales * 2 + i];
            featXPtsPerStep = shMem[numFeatScales * 3 + i];
            featYPtsPerStep = shMem[numFeatScales * 4 + i];
            nSubX = getNSubCurScale(exNx, featNx, featXPtsPerStep);
            nSubY = getNSubCurScale(exNy, featNy, featYPtsPerStep);
            nSub = nSubX * nSubY;
            
            getMinMaxIAndNextMinI(minIx, maxIx, nextMinIx, numFeatScales, prevLx, curShNx, featXPtsPerStep, exNx, featNx, i, true);
            getMinMaxIAndNextMinI(minIy, maxIy, nextMinIy, numFeatScales, prevLy, shNy, featYPtsPerStep, exNy, featNy, i, false);
            if (nextMinIx / featXPtsPerStep < nSubX) { 
                numFeatScalesX = i + 1; 
                xStep = min(nextMinIx - prevLx, xStep); 
            }
            if (minIx > maxIx || minIy > maxIy) { 
                cumNumSub += getNSubCurScale(exNx, featNx, featXPtsPerStep) * getNSubCurScale(exNy, featNy, featYPtsPerStep);
                continue;
            }

            curNSubY = (maxIy - minIy) / featYPtsPerStep + 1; 
            curNSub = ((maxIx - minIx) / featXPtsPerStep + 1) * curNSubY; 
        
            for (n = 0;;) {  

                j = n + threadIdx.x - m;

                if (j >= 0 && j < curNSub) {

                    subIx = minIx + (n + threadIdx.x - m) / curNSubY * featXPtsPerStep;
                    subIy = minIy + (n + threadIdx.x - m) % curNSubY * featYPtsPerStep;

                    if (isVolumetric) {distTh = INF;}
                    else {distTh = scale / 2;} 

                    if (coreId == 0) {
                        numValidTh = max(numValidTh, 3);
                        core_PCA_ISPRS12_gridded (featsPerSub, prevLx, prevLy, subIx, subIy, shNy, featNx, featNy, offsetShMem, 
                            initOffsetShMem, maxFloatInShMem, resolX, resolY, stdNormXY, stdNormZ, 
                            distTh, numValidTh);
                    } else if (coreId == 1) {
                        numValidTh = max(numValidTh, 3);
                        core_stats_gridded (featsPerSub, prevLx, prevLy, subIx, subIy, shNy, featNx, featNy, offsetShMem, 
                            initOffsetShMem, maxFloatInShMem, resolX, resolY, stdNormZ, randProps[blockIdx.x * blockDim.x + threadIdx.x], 
                            distTh, numValidTh);
                    } else {
                        //to do: in case we have more core methods in the future
                    }

                    for (j = 0; j < featDims; j++) {
                        feats[
                                blockIdx.x * featDims * nSubTotal 
                                + j * nSubTotal + (int)cumNumSub
                                + subIx / featXPtsPerStep * nSubY + subIy / featYPtsPerStep 
                            ] = featsPerSub[j];
                    }
                }
                
                if (n + blockDim.x - m >= curNSub) { 
                    m = (int)(m + ceil((curNSub - n) / 16.0) * 16) % blockDim.x;
                    break;
                } else {
                   n += blockDim.x - m; m = 0; 
                }
            }
            cumNumSub += nSub;

            if (!useMultiScale) {
                __syncthreads();
            }
            
        }
        __syncthreads();

        if (numFeatScalesX > 0) {  // still room to go right
            if (useIncreShmem) {
                offsetShMem = initOffsetShMem + (offsetShMem - initOffsetShMem + xStep * shNy) % (maxFloatInShMem - initOffsetShMem);
            } else {
                offsetShMem = initOffsetShMem;
            }         
            curNumFeatScales = numFeatScalesX;

            if (threadIdx.x < curNumFeatScales) {
                updateMinIInShMem(minIx, maxIx, nextMinIx, numFeatScales, prevLx, curShNx, 
                    shMem[numFeatScales * 3 + threadIdx.x], exNx, shMem[numFeatScales + threadIdx.x], threadIdx.x, true);
            }
            __syncthreads(); 

            prevShNx = curShNx;
            curShNx = min(shNx, exNx - (prevLx + xStep));

        } else { //no room to go right
            offsetShMem = initOffsetShMem; xStep = prevShNx = curShNx = shNx; prevLx = -shNx; 

            numFeatScalesY = 0; yStep = INF;
            for (i = 0; i < numFeatScales; i++) { 
                featNy = shMem[numFeatScales * 2 + i];
                featYPtsPerStep = shMem[numFeatScales * 4 + i];
                nSubY = getNSubCurScale(exNy, featNy, featYPtsPerStep);
                getMinMaxIAndNextMinI(minIy, maxIy, nextMinIy, numFeatScales, prevLy, shNy, featYPtsPerStep, exNy, featNy, i, false);
                if (nextMinIy / featYPtsPerStep < nSubY) { 
                    numFeatScalesY = i + 1;
                    yStep = min(nextMinIy - prevLy, yStep);
                }
            }
            if (numFeatScalesY == 0) { //cannot go up either, this is the end!
                break;
            }

            // go up
            curNumFeatScales = numFeatScalesY;
            if (threadIdx.x < curNumFeatScales) { 
                shMem[numFeatScales * 5 + threadIdx.x] = 0; 
                updateMinIInShMem(minIy, maxIy, nextMinIy, numFeatScales, prevLy, shNy, 
                    shMem[numFeatScales * 4 + threadIdx.x], exNy, shMem[numFeatScales * 2 + threadIdx.x], threadIdx.x, false);
            }
            __syncthreads();
            prevLy += yStep; 
            shNy = min(shNy, exNy - prevLy);
        }
    }
}

// parallel reduction
__device__ void updateSumInShMem(int& startSrc, int& startTgt, int& lenSrc) {

    int ind, i, numPerThread, lenTgt;
    double incre;

    numPerThread = max(ceil((double)lenSrc / blockDim.x), 2.0);
    lenTgt = ceil((double)lenSrc / numPerThread);

    if (threadIdx.x < lenTgt) {
        shMem[startTgt + threadIdx.x] = 0;
        for (i = 0; i < numPerThread; i++) {
            ind = threadIdx.x * numPerThread + i;
            if (ind >= lenSrc) {
                break;
            }
            shMem[startTgt + threadIdx.x] += shMem[startSrc + ind];
        }
    }
    swap(startSrc, startTgt);
    lenSrc = lenTgt;
}

extern "C" __global__ 
void aggregate_feats(float* aggFeats, float* sortedRawFeats, int* cumNumSubByScale, 
    int numFeatScales, int featDims, int numValidTh = 1) {
    
    int i, j, k, u, v, w, curBatchSize, numValid;
    double sums[4], avg, stdev, stat, gamma;  
    const int batchSize = 2 * blockDim.x;


    if (threadIdx.x <= numFeatScales) { 
        shMem[threadIdx.x] = cumNumSubByScale[threadIdx.x];
    }
    __syncthreads();

    u = numFeatScales * featDims;
    v = blockIdx.x % u;
    const int iEx = blockIdx.x / u;
    const int iDim = v / numFeatScales;
    const int iScale = v % numFeatScales;
    const int nSubTotal = shMem[numFeatScales];
    const int offset = iEx * nSubTotal * featDims + iDim * nSubTotal + (int)(shMem[iScale]);
    const int nFeat = shMem[iScale + 1] - shMem[iScale];
    const int nIter = ceil((double)nFeat / batchSize);
    __syncthreads();

   
    sums[0] = sums[1] = sums[2] = sums[3] = 0; //sums of powers
    stat = numValid = 0;
    for (i = nIter - 1; i >= 0; i--) {

        curBatchSize = min(batchSize, nFeat - i * batchSize);
        j = i * batchSize + threadIdx.x;
        if (threadIdx.x < curBatchSize) {
            shMem[threadIdx.x] = sortedRawFeats[offset + j];
        }
        j += blockDim.x;
        if (threadIdx.x + blockDim.x < curBatchSize) {
            shMem[blockDim.x + threadIdx.x] = sortedRawFeats[offset + j];
        }
        __syncthreads();

        if (numValid == 0) {
            if (shMem[0] == INF) {  //all invalid in this iteration
                continue;
            }

            for (; curBatchSize > 0; curBatchSize--) {
                if (shMem[curBatchSize - 1] != INF) {
                    break;
                }
            }
            numValid = i * batchSize + curBatchSize;

            if (numValid >= numValidTh && threadIdx.x >= 4 && threadIdx.x <= 8) {

                k = floor(1.0 / 3 + (numValid + 1.0 / 3) * (threadIdx.x - 4) / 4.0);

                if (threadIdx.x == 4 || k < 1) {  //go for minimum
                    k = 1; gamma = 0;
                } else if (threadIdx.x == 8 || k + 1 > numValid) { //go for maximum
                    k = numValid - 1; gamma = 1; 
                } else {
                    gamma = numValid * (threadIdx.x - 4) / 4.0 - k + ((threadIdx.x - 4) / 4.0 + 1) / 3;
                }
            }
        }

        if (numValid < numValidTh) {
            continue;
        }

        if (threadIdx.x >= 4 && threadIdx.x <= 8) {
            if (i == floor((double)(k - 1) / batchSize)) {
                j = (k - 1) % batchSize;

                stat += (1 - gamma) * shMem[j];
            }
            if (i == floor((double)k / batchSize)) {
                j = k % batchSize;

                stat += gamma * shMem[j];
            }
        }
        __syncthreads();
        
        for (j = 1; j <= 4; j++) { 
            
            if (threadIdx.x < curBatchSize) {  
                shMem[batchSize + threadIdx.x] = power(shMem[threadIdx.x], j);
            }
            if (threadIdx.x + blockDim.x < curBatchSize) {
                shMem[batchSize + threadIdx.x + blockDim.x] = power(shMem[threadIdx.x + blockDim.x], j); 
            }
            __syncthreads();

            u = batchSize; w = 2 * batchSize;
            v = curBatchSize;
            while (v != 1) {
                updateSumInShMem(u, w, v); 
                 __syncthreads();
            }
            sums[j - 1] += shMem[u]; 
        }
    }

    if (numValid >= numValidTh) {
        if (threadIdx.x < 4){
            if (threadIdx.x == 0) { //mean
                stat = (double)sums[0] / numValid;
            } else if (threadIdx.x == 1) {  //std
                stat = stdv(sums[0], sums[1], numValid);
            } else {
                avg = (double)sums[0] / numValid; stdev = stdv(sums[0], sums[1], numValid);
                if (threadIdx.x == 2) { //skewness
                    stat = skewness(avg, stdev, sums[0], sums[1], sums[2], numValid);
                } else {    //kurtosis
                    stat = kurtosis(avg, stdev, sums[0], sums[1], sums[2], sums[3], numValid);
                }
            }

        }
    } else {
        stat = INF;
    }
    if (threadIdx.x < 9) { //9 is the number of stats
        aggFeats[iEx * numFeatScales * featDims * 9 + iScale * featDims * 9 + iDim * 9 + threadIdx.x] = stat;
    }
}
