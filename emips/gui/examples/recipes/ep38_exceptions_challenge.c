/* MIPS by Example in eMIPS, episode 38 — the challenge, solved (the C version
 * of ep38_exceptions_challenge.asm): also a store to a bad address and a
 * break, each kind counted.  C has no word for an exception handler: emips.h's
 * EMIPS_EXCEPTION_HANDLER puts a short assembly entry at 0x80000180 that
 * saves the registers, calls on_exception and returns where it says. */
#include "emips.h"

static volatile int failed;            /* 1: the last try failed */
static int count;                      /* exceptions handled */
static int kinds[32];                  /* how many of each ExcCode */

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
    else if (code == 9)
        print_string("breakpoint");
    else if (code == 4 || code == 5) {
        print_string(code == 4 ? "bad address " : "bad store ");
        print_hex(badvaddr);
    } else
        print_string("unexpected");
    print_string("] ");
    failed = 1;
    count++;
    kinds[code]++;
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

static void try_store(unsigned address)
{
    print_string("store at ");
    print_hex(address);
    print_string(" = ");
    failed = 0;
    *(volatile int *)address = 7;       /* an odd address: Address Error (store) */
    result(7);
}

static void try_break(void)
{
    print_string("break = ");
    failed = 0;
    __asm__ volatile ("break 3");       /* Bp: a breakpoint */
    result(0);
}

static void summary(void)
{
    print_string("exceptions by kind:");
    for (int code = 0; code < 32; code++)
        if (kinds[code]) {
            print_char(' ');
            print_int(code);
            print_char('=');
            print_int(kinds[code]);
        }
    print_char('\n');
}

int main(void)
{
    try_add(7, 5);
    try_add(0x7fffffff, 1);
    try_div(100, 7);
    try_div(42, 0);
    try_load(0x10010001);
    try_store(0x10010002);
    try_break();
    summary();
    return 0;
}
