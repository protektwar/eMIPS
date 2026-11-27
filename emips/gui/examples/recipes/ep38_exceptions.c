/* MIPS by Example in eMIPS, episode 38 — Your own exception handler: an
 * overflow, a division by zero, a bad address (the C version of
 * ep38_exceptions.asm).  C has no word for an exception handler: emips.h's
 * EMIPS_EXCEPTION_HANDLER puts a short assembly entry at 0x80000180 that
 * saves the registers, calls on_exception and returns where it says. */
#include "emips.h"

static volatile int failed;            /* 1: the last try failed */
static int count;                      /* exceptions handled */

unsigned on_exception(unsigned cause, unsigned epc, unsigned badvaddr)
{
    unsigned code = (cause >> 2) & 31;  /* ExcCode, bits 6..2 */
    print_string("[exception ");
    print_int(code);
    print_string(": ");
    if (code == 12)
        print_string("overflow");
    else if (code == 13)
        print_string("trap (divide by zero)");
    else if (code == 4) {
        print_string("bad address ");
        print_hex(badvaddr);
    } else
        print_string("unexpected");
    print_string("] ");
    failed = 1;
    count++;
    return epc + 4;                     /* skip the instruction that failed */
}
EMIPS_EXCEPTION_HANDLER(on_exception);

static void result(int value)
{
    if (failed)
        print_string("skipped\n");
    else {
        print_int(value);
        print_char('\n');
    }
}

static void try_add(int a, int b)
{
    int sum;
    print_int(a);
    print_string(" + ");
    print_int(b);
    print_string(" = ");
    failed = 0;
    /* C's + never traps (GCC writes addu): add, the trapping one, by hand */
    __asm__ volatile ("add %0, %1, %2" : "=r"(sum) : "r"(a), "r"(b));
    result(sum);
}

static void try_div(int a, int b)
{
    print_int(a);
    print_string(" / ");
    print_int(b);
    print_string(" = ");
    failed = 0;
    result(a / b);                      /* GCC: teq b, $zero, 7 before div */
}

static void try_load(unsigned address)
{
    print_string("word at ");
    print_hex(address);
    print_string(" = ");
    failed = 0;
    result(*(volatile int *)address);   /* an odd address: Address Error */
}

int main(void)
{
    try_add(7, 5);
    try_add(0x7fffffff, 1);
    try_div(100, 7);
    try_div(42, 0);
    try_load(0x10010001);
    print_string("exceptions handled: ");
    print_int(count);
    print_char('\n');
    return 0;
}
