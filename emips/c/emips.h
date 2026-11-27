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

#ifdef __riscv
/* RISC-V (emips cc --arch rv32 / rv64): RARS's environment calls — a7 =
 * the service, a0 / a1 / fa0 the arguments, the result in a0 (RARS reads
 * the registers as 32-bit ints: print_long prints the low 32 bits). */
static inline long emips_ecall2(long service, long a0, long a1)
{
    register long a7_ __asm__("a7") = service;
    register long a0_ __asm__("a0") = a0;
    register long a1_ __asm__("a1") = a1;
    __asm__ volatile ("ecall" : "+r"(a0_) : "r"(a7_), "r"(a1_) : "memory");
    return a0_;
}
#define emips_syscall1(s, a) emips_ecall2((s), (long)(a), 0)
#define emips_syscall2(s, a, b) emips_ecall2((s), (long)(a), (long)(b))

static inline void print_int(int x)          { emips_syscall1(1, x); }
static inline void print_long(long x)        { emips_syscall1(1, x); }
static inline void print_string(const char *s) { emips_syscall1(4, (long)s); }
static inline void print_char(char c)        { emips_syscall1(11, c); }
static inline void print_hex(unsigned long x) { emips_syscall1(34, (long)x); }
static inline void print_bin(unsigned long x) { emips_syscall1(35, (long)x); }
static inline void print_unsigned(unsigned long x) { emips_syscall1(36, (long)x); }
static inline int  read_int(void)            { return emips_syscall1(5, 0); }
static inline long read_long(void)           { return (int)emips_syscall1(5, 0); }
static inline int  read_char(void)           { return emips_syscall1(12, 0); }
static inline void read_string(char *buf, int size) { emips_syscall2(8, (long)buf, size); }
static inline void *sbrk(int bytes)          { return (void *)emips_syscall1(9, bytes); }
static inline void exit(int code)            { emips_syscall1(93, code); for (;;) {} }
#ifdef __riscv_flen
static inline void print_float(float f)
{
    register float fa0_ __asm__("fa0") = f;
    register long a7_ __asm__("a7") = 2;
    __asm__ volatile ("ecall" : : "f"(fa0_), "r"(a7_) : "memory");
}
#endif
#if __riscv_flen >= 64
static inline void print_double(double d)
{
    register double fa0_ __asm__("fa0") = d;
    register long a7_ __asm__("a7") = 3;
    __asm__ volatile ("ecall" : : "f"(fa0_), "r"(a7_) : "memory");
}

static inline double read_double(void)
{
    register double fa0_ __asm__("fa0");
    register long a7_ __asm__("a7") = 7;
    __asm__ volatile ("ecall" : "=f"(fa0_) : "r"(a7_) : "memory");
    return fa0_;
}
#endif

/* RARS's dialogs (services 50-56): the answers come back in a0 / a1 */
static inline int input_dialog_int(const char *msg, int *status)
{
    register long a0_ __asm__("a0") = (long)msg;
    register long a1_ __asm__("a1");
    register long a7_ __asm__("a7") = 51;
    __asm__ volatile ("ecall" : "+r"(a0_), "=r"(a1_) : "r"(a7_) : "memory");
    *status = (int)a1_;                   /* 0, -1 not a number, -2 Cancel, -3 empty */
    return (int)a0_;
}

static inline int confirm_dialog(const char *msg)  /* 0 Yes, 1 No, 2 Cancel */
{
    register long a0_ __asm__("a0") = (long)msg;
    register long a7_ __asm__("a7") = 50;
    __asm__ volatile ("ecall" : "+r"(a0_) : "r"(a7_) : "memory");
    return (int)a0_;
}

static inline void message_dialog(const char *msg, int type) { emips_ecall2(55, (long)msg, type); }
static inline void message_dialog_int(const char *msg, int n) { emips_ecall2(56, (long)msg, n); }

/* RARS's time and random numbers (30, 40, 42) */
static inline unsigned time_ms(void)
{
    register long a0_ __asm__("a0");
    register long a1_ __asm__("a1");
    register long a7_ __asm__("a7") = 30;
    __asm__ volatile ("ecall" : "=r"(a0_), "=r"(a1_) : "r"(a7_) : "memory");
    return (unsigned)a0_;
}

static inline void set_seed(int stream, int seed) { emips_ecall2(40, stream, seed); }

static inline int random_int_range(int stream, int bound)
{
    register long a0_ __asm__("a0") = stream;
    register long a1_ __asm__("a1") = bound;
    register long a7_ __asm__("a7") = 42;
    __asm__ volatile ("ecall" : "+r"(a0_) : "r"(a1_), "r"(a7_) : "memory");
    return (int)a0_;
}

/* RARS's files (1024 open, 63 read, 64 write, 57 close) */
static inline long emips_file3(long service, long a, long b, long c)
{
    register long a0_ __asm__("a0") = a;
    register long a1_ __asm__("a1") = b;
    register long a2_ __asm__("a2") = c;
    register long a7_ __asm__("a7") = service;
    __asm__ volatile ("ecall" : "+r"(a0_) : "r"(a1_), "r"(a2_), "r"(a7_) : "memory");
    return a0_;
}
/* RARS's MIDI out (31, 33: a0 pitch, a1 ms, a2 instrument, a3 volume) */
static inline void emips_midi(long service, int pitch, int ms, int instrument, int volume)
{
    register long a0_ __asm__("a0") = pitch;
    register long a1_ __asm__("a1") = ms;
    register long a2_ __asm__("a2") = instrument;
    register long a3_ __asm__("a3") = volume;
    register long a7_ __asm__("a7") = service;
    __asm__ volatile ("ecall" : : "r"(a0_), "r"(a1_), "r"(a2_), "r"(a3_), "r"(a7_) : "memory");
}
static inline void midi_out(int pitch, int ms, int instrument, int volume)
{ emips_midi(31, pitch, ms, instrument, volume); }
static inline void midi_out_sync(int pitch, int ms, int instrument, int volume)
{ emips_midi(33, pitch, ms, instrument, volume); }

static inline int open_file(const char *name, int flags) { return emips_file3(1024, (long)name, flags, 0); }
static inline int read_file(int fd, void *buf, int n) { return emips_file3(63, fd, (long)buf, n); }
static inline int write_file(int fd, const void *buf, int n) { return emips_file3(64, fd, (long)buf, n); }
static inline void close_file(int fd) { emips_file3(57, fd, 0, 0); }

/* The MMIO console (RARS = MARS: keyboard and display at 0xFFFF0000). */
#define EMIPS_IO(a)     (*(volatile unsigned *)(unsigned long)(a))
#define MMIO_RX_CONTROL EMIPS_IO(0xFFFF0000u)
#define MMIO_RX_DATA    EMIPS_IO(0xFFFF0004u)
#define MMIO_TX_CONTROL EMIPS_IO(0xFFFF0008u)
#define MMIO_TX_DATA    EMIPS_IO(0xFFFF000Cu)

/* CSRs from C (M-mode): csrr / csrw of a CSR by name (CSRR(mstatus)). */
#define CSRR(csr) ({ unsigned long _v; __asm__ volatile ("csrr %0, " #csr : "=r"(_v)); _v; })
#define CSRW(csr, v) __asm__ volatile ("csrw " #csr ", %0" : : "r"((unsigned long)(v)))

typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

#else   /* MIPS */

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
#ifdef __mips_hard_float
EMIPS_MIPS32 void print_double(double d)
{
    __asm__ volatile ("mov.d $f12, %0\n\tli $v0, 3\n\tsyscall" : : "f"(d)
                      : "$2", "$f12", "$f13", "memory");
}

EMIPS_MIPS32 double read_double(void)
{
    double d;
    __asm__ volatile ("li $v0, 7\n\tsyscall\n\tmov.d %0, $f0" : "=f"(d) : : "$2", "$f0", "$f1",
                      "memory");
    return d;
}
#endif

/* MARS's dialogs (services 50-59): the answers come back in $a0 / $a1 */
EMIPS_MIPS32 int input_dialog_int(const char *msg, int *status)
{
    register long v0 __asm__("$2") = 51;
    register long r4 __asm__("$4") = (long)msg;
    register long r5 __asm__("$5");
    __asm__ volatile ("syscall" : "+r"(v0), "+r"(r4), "=r"(r5) : : "memory");
    *status = (int)r5;                    /* 0, -1 not a number, -2 Cancel, -3 empty */
    return (int)r4;
}

EMIPS_MIPS32 int confirm_dialog(const char *msg)   /* 0 Yes, 1 No, 2 Cancel */
{
    register long v0 __asm__("$2") = 50;
    register long r4 __asm__("$4") = (long)msg;
    __asm__ volatile ("syscall" : "+r"(v0), "+r"(r4) : : "memory");
    return (int)r4;
}

static inline void message_dialog(const char *msg, int type) { emips_syscall2(55, (long)msg, type); }
static inline void message_dialog_int(const char *msg, int n) { emips_syscall2(56, (long)msg, n); }

/* MARS's time and random numbers (30, 40, 42): in eMIPS the time is the
 * simulated clock, so a run replays exactly */
EMIPS_MIPS32 unsigned time_ms(void)       /* the low word of the milliseconds */
{
    register long v0 __asm__("$2") = 30;
    register long r4 __asm__("$4");
    register long r5 __asm__("$5");
    __asm__ volatile ("syscall" : "+r"(v0), "=r"(r4), "=r"(r5) : : "memory");
    return (unsigned)r4;
}

static inline void set_seed(int stream, int seed) { emips_syscall2(40, stream, seed); }

EMIPS_MIPS32 int random_int_range(int stream, int bound)   /* 0 ... bound - 1 */
{
    register long v0 __asm__("$2") = 42;
    register long r4 __asm__("$4") = stream;
    register long r5 __asm__("$5") = bound;
    __asm__ volatile ("syscall" : "+r"(v0), "+r"(r4) : "r"(r5) : "memory");
    return (int)r4;
}

/* MARS's files (13-16): the files given with --file; flags 0 read, 1 write,
 * 9 append; a result below 0 is an error */
EMIPS_MIPS32 long emips_file3(long service, long a, long b, long c)
{
    register long v0 __asm__("$2") = service;
    register long r4 __asm__("$4") = a;
    register long r5 __asm__("$5") = b;
    register long r6 __asm__("$6") = c;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4), "r"(r5), "r"(r6) : "memory");
    return v0;
}
/* MARS's MIDI out: 31 plays a note and goes on, 33 waits until it ends
 * (pitch 60 = middle C, milliseconds, instrument 0-127, volume 0-127) */
EMIPS_MIPS32 void emips_midi(long service, int pitch, int ms, int instrument, int volume)
{
    register long v0 __asm__("$2") = service;
    register long r4 __asm__("$4") = pitch;
    register long r5 __asm__("$5") = ms;
    register long r6 __asm__("$6") = instrument;
    register long r7 __asm__("$7") = volume;
    __asm__ volatile ("syscall" : "+r"(v0) : "r"(r4), "r"(r5), "r"(r6), "r"(r7) : "memory");
}
static inline void midi_out(int pitch, int ms, int instrument, int volume)
{ emips_midi(31, pitch, ms, instrument, volume); }
static inline void midi_out_sync(int pitch, int ms, int instrument, int volume)
{ emips_midi(33, pitch, ms, instrument, volume); }

static inline int open_file(const char *name, int flags) { return emips_file3(13, (long)name, flags, 0); }
static inline int read_file(int fd, void *buf, int n) { return emips_file3(14, fd, (long)buf, n); }
static inline int write_file(int fd, const void *buf, int n) { return emips_file3(15, fd, (long)buf, n); }
static inline void close_file(int fd) { emips_file3(16, fd, 0, 0); }

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

#if defined(_ABIO32) && _MIPS_SIM == _ABIO32 && !defined(__nanomips__)
/* An exception handler in C (o32): EMIPS_EXCEPTION_HANDLER(fn) puts the
 * entry at the general exception vector, 0x80000180 (.ktext: emips.ld).  It
 * saves the registers C code may change ($at, $v0-$v1, $a0-$a3, $t0-$t9,
 * $ra, HI, LO) on the interrupted code's stack, calls
 *     unsigned fn(unsigned cause, unsigned epc, unsigned badvaddr)
 * (a function of yours, not static; write EMIPS_EXCEPTION_HANDLER(fn); at
 * file scope), puts what it returns in EPC — where
 * the program goes on: EPC + 4 skips the instruction that failed, EPC itself
 * comes back to an interrupted one —, restores them and returns with ERET.
 * An exception in a branch delay slot (Cause.BD, bit 31) has the branch in
 * EPC: skipping it is not that simple, keep faulting code out of delay slots.
 * The handler runs with Status.EXL = 1: interrupts are off, system calls
 * (print_string ...) work. */
#define EMIPS_EXCEPTION_HANDLER(fn) \
__asm__(".pushsection .ktext, \"ax\", @progbits\n" \
"\t.set  push\n\t.set  noreorder\n\t.set  noat\n\t.set  nomips16\n" \
"\t.globl emips_exception\n" \
"emips_exception:\n" \
"\taddiu $sp, $sp, -96\n" \
"\tsw    $at, 16($sp)\n" \
"\tsw    $v0, 20($sp)\n" \
"\tsw    $v1, 24($sp)\n" \
"\tsw    $a0, 28($sp)\n" \
"\tsw    $a1, 32($sp)\n" \
"\tsw    $a2, 36($sp)\n" \
"\tsw    $a3, 40($sp)\n" \
"\tsw    $t0, 44($sp)\n" \
"\tsw    $t1, 48($sp)\n" \
"\tsw    $t2, 52($sp)\n" \
"\tsw    $t3, 56($sp)\n" \
"\tsw    $t4, 60($sp)\n" \
"\tsw    $t5, 64($sp)\n" \
"\tsw    $t6, 68($sp)\n" \
"\tsw    $t7, 72($sp)\n" \
"\tsw    $t8, 76($sp)\n" \
"\tsw    $t9, 80($sp)\n" \
"\tsw    $ra, 84($sp)\n" \
"\tmfhi  $k0\n\tsw    $k0, 88($sp)\n\tmflo  $k0\n\tsw    $k0, 92($sp)\n" \
"\tmfc0  $a0, $13\n\tmfc0  $a1, $14\n\tmfc0  $a2, $8\n" \
"\tlui   $t9, %hi(" #fn ")\n\taddiu $t9, $t9, %lo(" #fn ")\n" \
"\tjalr  $t9\n\tnop\n" \
"\tmtc0  $v0, $14\n" \
"\tlw    $k0, 88($sp)\n\tmthi  $k0\n\tlw    $k0, 92($sp)\n\tmtlo  $k0\n" \
"\tlw    $at, 16($sp)\n" \
"\tlw    $v0, 20($sp)\n" \
"\tlw    $v1, 24($sp)\n" \
"\tlw    $a0, 28($sp)\n" \
"\tlw    $a1, 32($sp)\n" \
"\tlw    $a2, 36($sp)\n" \
"\tlw    $a3, 40($sp)\n" \
"\tlw    $t0, 44($sp)\n" \
"\tlw    $t1, 48($sp)\n" \
"\tlw    $t2, 52($sp)\n" \
"\tlw    $t3, 56($sp)\n" \
"\tlw    $t4, 60($sp)\n" \
"\tlw    $t5, 64($sp)\n" \
"\tlw    $t6, 68($sp)\n" \
"\tlw    $t7, 72($sp)\n" \
"\tlw    $t8, 76($sp)\n" \
"\tlw    $t9, 80($sp)\n" \
"\tlw    $ra, 84($sp)\n" \
"\taddiu $sp, $sp, 96\n" \
"\tehb\n\teret\n" \
"\t.set  pop\n\t.popsection\n")
#endif

typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
void *memmove(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

#endif  /* MIPS */
#endif
