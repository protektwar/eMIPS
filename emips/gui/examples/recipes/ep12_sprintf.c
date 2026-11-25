/* MIPS by Example in eMIPS, episode 12 — A tiny sprintf (the C version of
 * ep12_sprintf.asm): %d, %s, %c and %% with C's variadic arguments. */
#include <stdarg.h>
#include "emips.h"

static char buf[64];

static char *put_dec(char *p, int n)      /* n in decimal at p; the end */
{
    unsigned u = n < 0 ? 0u - (unsigned)n : (unsigned)n, t = u;
    char *end;

    if (n < 0)
        *p++ = '-';
    do {                                  /* count the digits ... */
        p++;
        t /= 10;
    } while (t != 0);
    end = p;
    do {                                  /* ... then write them from the right */
        *--p = '0' + u % 10;
        u /= 10;
    } while (u != 0);
    return end;
}

int tiny_sprintf(char *out, const char *fmt, ...)
{
    va_list args;
    char *p = out;

    va_start(args, fmt);
    for (; *fmt != '\0'; fmt++) {
        if (*fmt != '%') {
            *p++ = *fmt;
            continue;
        }
        switch (*++fmt) {
        case 'd':
            p = put_dec(p, va_arg(args, int));
            break;
        case 's':
            for (const char *s = va_arg(args, const char *); *s != '\0'; s++)
                *p++ = *s;
            break;
        case 'c':
            *p++ = (char)va_arg(args, int);
            break;
        default:                          /* %% (or anything else): itself */
            *p++ = *fmt;
        }
    }
    va_end(args);
    *p = '\0';
    return p - out;
}

static void show(int n)
{
    print_string(buf);
    print_string(" (");
    print_int(n);
    print_string(" characters)\n");
}

int main(void)
{
    show(tiny_sprintf(buf, "%s is %d years old, grade %c%%", "Ada", 36, 'A'));
    show(tiny_sprintf(buf, "%d + %d = %d", 2, 40, 42));
    return 0;
}
