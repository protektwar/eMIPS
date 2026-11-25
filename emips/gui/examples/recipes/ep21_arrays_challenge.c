/* MIPS by Example in eMIPS, episode 21, the challenge solved (the C version
 * of ep21_arrays_challenge.asm). */
#include "emips.h"

static const int midterm[6] = {78, 92, 65, 88, 71, 95};
static const int final[6] = {84, 88, 59, 92, 75, 99};
static const char letters[6] = {'B', 'A', 'D', 'A', 'C', 'A'};
static int counts[5];                     /* A B C D F */
static const char names[5] = {'A', 'B', 'C', 'D', 'F'};

int best(const int *m, const int *f, int n, int *avg)
{
    int b = 0, bavg = -1;
    for (int i = 0; i < n; i++) {
        int a = (unsigned)(m[i] + f[i]) / 2;
        if (a > bavg) {                   /* a tie keeps the first */
            b = i;
            bavg = a;
        }
    }
    *avg = bavg;
    return b;
}

void count_grades(const char *l, int n, int *c)
{
    for (int i = 0; i < n; i++) {
        int k = l[i] - 'A';
        if (k >= 5)                       /* F: the fifth counter */
            k = 4;
        c[k]++;
    }
}

int main(void)
{
    int avg, b = best(midterm, final, 6, &avg);

    print_string("best: student ");
    print_int(b);
    print_string(", average ");
    print_int(avg);
    print_char('\n');
    count_grades(letters, 6, counts);
    for (int i = 0; i < 5; i++) {
        print_char(names[i]);
        print_char(' ');
        print_int(counts[i]);
        if (i < 4)
            print_string(", ");
    }
    print_char('\n');
    return 0;
}
