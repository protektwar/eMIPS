/* emips.h — talking to eMIPS from C (python3 -m emips cc).
 *
 * There is no C library: these call the MARS / SPIM system call services
 * eMIPS provides ($v0 = the service, $a0.. the arguments, then SYSCALL),
 * and reach the MMIO console at 0xFFFF0000.  Everything is inline.
 *
 * o32 and n64 (MIPS64, emips cc --arch mips64): the arguments go as whole
 * registers (long: 32 / 64 bits) — on a MIPS64 core the services print and
 * read 64-bit values (print_long, read_long, print_hex of 16 digits). */
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

EMIPS_MIPS32 long emips_syscall1(long service, long a0)
{
    register long v0 __asm__("$2") = service;
    register long r4 __asm__("$4") = a0;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4) : "memory");
    return v0;
}

EMIPS_MIPS32 long emips_syscall2(long service, long a0, long a1)
{
    register long v0 __asm__("$2") = service;
    register long r4 __asm__("$4") = a0;
    register long r5 __asm__("$5") = a1;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4), "r"(r5) : "memory");
    return v0;
}

static inline void print_int(int x)          { emips_syscall1(1, x); }
static inline void print_long(long x)        { emips_syscall1(1, x); }
static inline void print_string(const char *s) { emips_syscall1(4, (long)s); }
static inline void print_char(char c)        { emips_syscall1(11, c); }
static inline void print_hex(unsigned long x) { emips_syscall1(34, (long)x); }
static inline void print_bin(unsigned long x) { emips_syscall1(35, (long)x); }
static inline void print_unsigned(unsigned long x) { emips_syscall1(36, (long)x); }
static inline int  read_int(void)            { return emips_syscall1(5, 0); }
static inline long read_long(void)           { return emips_syscall1(5, 0); }
static inline int  read_char(void)           { return emips_syscall1(12, 0); }
static inline void read_string(char *buf, int size) { emips_syscall2(8, (long)buf, size); }
static inline void *sbrk(int bytes)          { return (void *)emips_syscall1(9, bytes); }
static inline void exit(int code)            { emips_syscall1(17, code); for (;;) {} }

EMIPS_MIPS32 void print_float(float f)
{
    __asm__ volatile ("mov.s $f12, %0\n\tli $v0, 2\n\tsyscall" : : "f"(f) : "$2", "$f12", "memory");
}

/* The MMIO console (MARS "Keyboard and Display MMIO", kseg3: kernel mode).
 * (long)(int): a kseg3 address sign-extended, as n64 pointers need it. */
#define EMIPS_IO(a)     (*(volatile unsigned *)(long)(int)(a))
#define MMIO_RX_CONTROL EMIPS_IO(0xFFFF0000u)
#define MMIO_RX_DATA    EMIPS_IO(0xFFFF0004u)
#define MMIO_TX_CONTROL EMIPS_IO(0xFFFF0008u)
#define MMIO_TX_DATA    EMIPS_IO(0xFFFF000Cu)

/* CP0 from C (kernel mode): mfc0 / mtc0 of register r, select s. */
#define MFC0(r, s) ({ unsigned _v; __asm__ volatile ("mfc0 %0, $" #r ", " #s : "=r"(_v)); _v; })
#define MTC0(r, s, v) __asm__ volatile ("mtc0 %0, $" #r ", " #s "\n\tehb" : : "r"(v))

typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

#endif
