/* MIPS by Example in eMIPS, episode 12, the challenge solved (the C version
 * of ep12_sprintf_challenge.asm): %x and a one-digit width. */
#include <stdarg.h>
#include "emips.h"

static char buf[64];

static char *put_dec(char *p, int n)
{
    unsigned u = n < 0 ? 0u - (unsigned)n : (unsigned)n, t = u;
    char *end;

    if (n < 0)
        *p++ = '-';
    do {
        p++;
        t /= 10;
    } while (t != 0);
    end = p;
    do {
        *--p = '0' + u % 10;
        u /= 10;
    } while (u != 0);
    return end;
}

static char *put_hex(char *p, unsigned n)  /* no leading zeros */
{
    int started = 0;
    for (int shift = 28; shift >= 0; shift -= 4) {
        unsigned d = (n >> shift) & 0xf;
        if (d != 0 || started || shift == 0) {
            *p++ = "0123456789abcdef"[d];
            started = 1;
        }
    }
    return p;
}

static char *pad(char *start, char *end, int width)  /* right-aligned */
{
    int spaces = width - (end - start);
    if (spaces <= 0)
        return end;
    for (char *q = end; q != start; q--)
        q[spaces - 1] = q[-1];
    for (int i = 0; i < spaces; i++)
        start[i] = ' ';
    return end + spaces;
}

int tiny_sprintf(char *out, const char *fmt, ...)
{
    va_list args;
    char *p = out;

    va_start(args, fmt);
    for (; *fmt != '\0'; fmt++) {
        int width = 0;
        char *start;

        if (*fmt != '%') {
            *p++ = *fmt;
            continue;
        }
        fmt++;
        if ((unsigned)(*fmt - '0') < 10)
            width = *fmt++ - '0';
        start = p;
        switch (*fmt) {
        case 'd':
            p = put_dec(p, va_arg(args, int));
            break;
        case 'x':
            p = put_hex(p, va_arg(args, unsigned));
            break;
        case 's':
            for (const char *s = va_arg(args, const char *); *s != '\0'; s++)
                *p++ = *s;
            break;
        case 'c':
            *p++ = (char)va_arg(args, int);
            break;
        default:
            *p++ = *fmt;
            continue;
        }
        p = pad(start, p, width);
    }
    va_end(args);
    *p = '\0';
    return p - out;
}

int main(void)
{
    tiny_sprintf(buf, "[%5d] [%x] [%3s] [%c]", 42, 255, "ab", '!');
    print_string(buf);
    print_char('\n');
    return 0;
}
