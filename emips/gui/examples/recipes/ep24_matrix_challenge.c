/* MIPS by Example in eMIPS, episode 24, the challenge solved (the C version
 * of ep24_matrix_challenge.asm). */
#include "emips.h"

#define N 3

static int ma[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

void transpose(int a[N][N])
{
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {   /* above the diagonal only */
            int t = a[i][j];
            a[i][j] = a[j][i];
            a[j][i] = t;
        }
}

int trace(const int a[N][N])
{
    int sum = 0;
    for (int i = 0; i < N; i++)
        sum += a[i][i];
    return sum;
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
    transpose(ma);
    print_matrix("transposed:\n", ma);
    print_string("trace ");
    print_int(trace(ma));
    print_char('\n');
    return 0;
}
