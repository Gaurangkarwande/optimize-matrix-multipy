#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>

#define BLOCK_SIZE 33
#define min(a,b) (((a)<(b))?(a):(b))

void initialize_matrix(int row, int col, double (*mat)[col], int is_result)
{
    srand(0);
    int i, j;
    for ( i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            if (is_result) {
                mat[i][j] = 0;
            } else {
                mat[i][j] = rand() % (row*col);
            } 
        }
    }
   return; 
}

void display_matrix(int row, int col, int (*mat)[col])
{
    printf("\n");
    int i, j;
    for ( i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            printf("%d\t", mat[2][1]);
        }
        printf("\n");
    }
   return; 
}

void multiply_cblas(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, r1, c2, c1, alpha, &first[0][0], c1, &second[0][0], c2, beta, &result[0][0], c2);
}

void multiply_block(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    int ii, jj, kk, i, j, k, blk_size = BLOCK_SIZE;

	for(ii = 0; ii < r1; ii += blk_size) {
        for(kk = 0; kk < c1; kk += blk_size) {
            for(jj = 0; jj < c2; jj += blk_size) {
				for(i = ii; i < min(r1, ii+blk_size); i++) {
                    for(k = kk; k < min(c1, kk+blk_size); k++) {
                        for(j = jj; j < min(c2, jj+blk_size); j++) {
							result[i][j] += first[i][k] * second[k][j]; 
                        }
                    }
                }
            }
        }
    }											
}

void multiply_block_reuse(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    int ii, jj, kk, i, j, k, first_ik, blk_size = BLOCK_SIZE;

	for(ii = 0; ii < r1; ii += blk_size) {
        for(kk = 0; kk < c1; kk += blk_size) {
            for(jj = 0; jj < c2; jj += blk_size) {
				for(i = ii; i < min(r1, ii+blk_size); i++) {
                    for(k = kk; k < min(c1, kk+blk_size); k++) {
                        first_ik = first[i][k];
                        for(j = jj; j < min(c2, jj+blk_size); j++) {
							result[i][j] += first_ik * second[k][j]; 
                        }
                    }
                }
            }
        }
    }											
}

void multiply_block_unroll_reuse(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    int ii, jj, kk, i, j, k, first_ik, blk_size = BLOCK_SIZE;

	for(ii = 0; ii < r1; ii += blk_size) {
        for(kk = 0; kk < c1; kk += blk_size) {
            for(jj = 0; jj < c2; jj += blk_size) {
				for(i = ii; i < min(r1, ii+blk_size); i++) {
                    for(k = kk; k < min(c1, kk+blk_size); k++) {
                        first_ik = first[i][k];
                        for(j = jj; j < min(c2, jj+blk_size); j+=8) {
							if (min(c2, jj+blk_size) - j >= 8) 
                            {
                                result[i][j+0] += first_ik * second[k][j+0];
                                result[i][j+1] += first_ik * second[k][j+1];
                                result[i][j+2] += first_ik * second[k][j+2];
                                result[i][j+3] += first_ik * second[k][j+3];
                                result[i][j+4] += first_ik * second[k][j+4];
                                result[i][j+5] += first_ik * second[k][j+5];
                                result[i][j+6] += first_ik * second[k][j+6];
                                result[i][j+7] += first_ik * second[k][j+7];
                            }
                            else
                            {
                                for (; j<min(c2, jj+blk_size); j++) {
                                    result[i][j] += first_ik * second[k][j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }											
}

void multiply_unroll(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    for (int i = 0; i < r1; ++i) {
        for (int k = 0; k < c1; ++k) {
            for (int j = 0; j < c2; j+=8) {
                if (c2 - j >= 8) 
                {
                    result[i][j+0] += first[i][k] * second[k][j+0];
                    result[i][j+1] += first[i][k] * second[k][j+1];
                    result[i][j+2] += first[i][k] * second[k][j+2];
                    result[i][j+3] += first[i][k] * second[k][j+3];
                    result[i][j+4] += first[i][k] * second[k][j+4];
                    result[i][j+5] += first[i][k] * second[k][j+5];
                    result[i][j+6] += first[i][k] * second[k][j+6];
                    result[i][j+7] += first[i][k] * second[k][j+7];
                }
                else
                {
                    for (; j<c2; j++) {
                        result[i][j] += first[i][k] * second[k][j];
                    }
                }
            }
        }
    }
}

void multiply_unroll_reuse(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    int first_ik;
    for (int i = 0; i < r1; ++i) {
        for (int k = 0; k < c1; ++k) {
            first_ik = first[i][k];
            for (int j = 0; j < c2; j+=8) {
                if (c2 - j >= 8) 
                {
                    result[i][j+0] += first_ik * second[k][j+0];
                    result[i][j+1] += first_ik * second[k][j+1];
                    result[i][j+2] += first_ik * second[k][j+2];
                    result[i][j+3] += first_ik * second[k][j+3];
                    result[i][j+4] += first_ik * second[k][j+4];
                    result[i][j+5] += first_ik * second[k][j+5];
                    result[i][j+6] += first_ik * second[k][j+6];
                    result[i][j+7] += first_ik * second[k][j+7];
                }
                else
                {
                    for (; j<c2; j++) {
                        result[i][j] += first[i][k] * second[k][j];
                    }
                }
            }
        }
    }
}

void multiply_reorder_reuse(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    int first_ik;
    for (int i = 0; i < r1; ++i) {
        for (int k = 0; k < c1; ++k) {
            first_ik = first[i][k];
            for (int j = 0; j < c2; ++j) {
                result[i][j] += first_ik * second[k][j];
            }
        }
    }
}

void multiply_naive(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2], double (*result)[c2])
{
    for (int i = 0; i < r1; ++i) {        
        for (int j = 0; j < c2; ++j) {
            for (int k = 0; k < c1; ++k) {
                result[i][j] += first[i][k] * second[k][j];
            }
        }
    }
}

int compare_matrix(int r1, int c1, double (*first)[c1], int r2, int c2, double (*second)[c2])
{
    for (int i = 0; i < r1; ++i)
        for (int j = 0; j < c2; ++j) 
            if (first[i][j] != second[i][j])
                return -1;
    return 0;
}

int main()
{
    int dim = 500;
    static struct timeval str, end;
    unsigned long long time;
    double A[dim][dim], B[dim][dim], C[dim][dim], Baseline[dim][dim];
    initialize_matrix(dim, dim, A, 0);
    initialize_matrix(dim, dim, B, 0);
    initialize_matrix(dim, dim, C, 1);
    initialize_matrix(dim, dim, Baseline, 1);

    multiply_naive(dim, dim, A, dim, dim, B, Baseline);

    gettimeofday(&str, NULL);
    multiply_block_unroll_reuse(dim, dim, A, dim, dim, B, C);
    gettimeofday(&end, NULL);

    if (compare_matrix(dim, dim, Baseline, dim, dim, C) == -1)
    {
        printf("Matrix multiplication is wrong\n");
        return -1;
    }

    time = 1000 * (end.tv_sec - str.tv_sec) + (end.tv_usec - str.tv_usec) / 1000;
    printf("\n===================== \n");
    printf("Time elapsed: %llu ms \n", time);
    printf("===================== \n\n");

    return 0;
}