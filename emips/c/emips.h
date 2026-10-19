/* emips.h — talking to eMIPS from C (python3 -m emips cc).
 *
 * There is no C library: these call the MARS / SPIM system call services
 * eMIPS provides ($v0 = the service, $a0.. the arguments, then SYSCALL),
 * and reach the MMIO console at 0xFFFF0000.  Everything is inline. */
#ifndef EMIPS_H
#define EMIPS_H

/* MIPS16e code (gcc -mips16, emips cc --mips16) has no SYSCALL, MFC0 or
 * MTC0: the functions that use them are MIPS32 ones (nomips16), called
 * with JALX.  A function of yours that uses MFC0 / MTC0 needs
 * __attribute__((nomips16)) in a MIPS16e build too. */
#ifdef __mips16
#define EMIPS_MIPS32 static __attribute__((nomips16, noinline))
#else
#define EMIPS_MIPS32 static inline
#endif

EMIPS_MIPS32 int emips_syscall1(int service, int a0)
{
    register int v0 __asm__("$2") = service;
    register int r4 __asm__("$4") = a0;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4) : "memory");
    return v0;
}

EMIPS_MIPS32 int emips_syscall2(int service, int a0, int a1)
{
    register int v0 __asm__("$2") = service;
    register int r4 __asm__("$4") = a0;
    register int r5 __asm__("$5") = a1;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4), "r"(r5) : "memory");
    return v0;
}

static inline void print_int(int x)          { emips_syscall1(1, x); }
static inline void print_string(const char *s) { emips_syscall1(4, (int)s); }
static inline void print_char(char c)        { emips_syscall1(11, c); }
static inline void print_hex(unsigned x)     { emips_syscall1(34, (int)x); }
static inline void print_bin(unsigned x)     { emips_syscall1(35, (int)x); }
static inline void print_unsigned(unsigned x) { emips_syscall1(36, (int)x); }
static inline int  read_int(void)            { return emips_syscall1(5, 0); }
static inline int  read_char(void)           { return emips_syscall1(12, 0); }
static inline void read_string(char *buf, int size) { emips_syscall2(8, (int)buf, size); }
static inline void *sbrk(int bytes)          { return (void *)emips_syscall1(9, bytes); }
static inline void exit(int code)            { emips_syscall1(17, code); for (;;) {} }

EMIPS_MIPS32 void print_float(float f)
{
    __asm__ volatile ("mov.s $f12, %0\n\tli $v0, 2\n\tsyscall" : : "f"(f) : "$2", "$f12", "memory");
}

/* The MMIO console (MARS "Keyboard and Display MMIO", kseg3: kernel mode). */
#define MMIO_RX_CONTROL (*(volatile unsigned *)0xFFFF0000)
#define MMIO_RX_DATA    (*(volatile unsigned *)0xFFFF0004)
#define MMIO_TX_CONTROL (*(volatile unsigned *)0xFFFF0008)
#define MMIO_TX_DATA    (*(volatile unsigned *)0xFFFF000C)

/* CP0 from C (kernel mode): mfc0 / mtc0 of register r, select s. */
#define MFC0(r, s) ({ unsigned _v; __asm__ volatile ("mfc0 %0, $" #r ", " #s : "=r"(_v)); _v; })
#define MTC0(r, s, v) __asm__ volatile ("mtc0 %0, $" #r ", " #s "\n\tehb" : : "r"(v))

typedef unsigned int size_t;
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

#endif
