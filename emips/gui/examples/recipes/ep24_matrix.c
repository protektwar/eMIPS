/* MIPS by Example in eMIPS, episode 24 — Matrices: add and multiply (the C
 * version of ep24_matrix.asm). */
#include "emips.h"

#define N 3

static const int ma[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
static const int mb[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
static int mc[N][N];

void madd(const int a[N][N], const int b[N][N], int c[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            c[i][j] = a[i][j] + b[i][j];
}

void mmul(const int a[N][N], const int b[N][N], int c[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i][k] * b[k][j];  /* a row of A, a column of B */
            c[i][j] = sum;
        }
}

static void print_matrix(const char *text, const int m[N][N])
{
    print_string(text);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            print_int(m[i][j]);
            print_char(j == N - 1 ? '\n' : ' ');
        }
}

int main(void)
{
    madd(ma, mb, mc);
    print_matrix("A + B:\n", mc);
    mmul(ma, mb, mc);
    print_matrix("A x B:\n", mc);
    return 0;
}
