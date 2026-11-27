/* crt0.h — crt0.S's macros (python3 -m emips cc), for o32 (MIPS32) and
 * n64 (MIPS64).  (In a header so that crt0.S's lines stay where the
 * prebuilt executables' line tables say they are.) */
#if _MIPS_SZPTR == 64
#define PTR_ADDIU daddiu        /* n64: 64-bit pointers */
#define PTR_SUBU dsubu
#else
#define PTR_ADDIU addiu
#define PTR_SUBU subu
#endif

/* An address: lui + addiu, the 32-bit sum sign-extended.  (n64's la with
 * -msym32 is lui + daddiu: for __stack_top = 0x7FFFEFF0, %hi = 0x8000 and
 * the 64-bit sum is 0xFFFFFFFF7FFFEFF0.) */
#define LA(reg, sym) lui reg, %hi(sym); addiu reg, reg, %lo(sym)

/* The stack: __stack_top, less what the boot set this core's $sp below its
 * own top, in 64 KB steps — on a CPS each core's $sp starts 64 KB lower
 * (emips.session CORE_STACK), and every core runs crt0: one stack for all
 * would have them overwrite each other's locals.  A reset boot leaves $sp 0:
 * __stack_top.  One line in crt0.S, two words as the LA it replaced (a call
 * to __emips_stack, which emips.ld puts after all the code: the program's
 * addresses do not move).  There: $t1 = the boot's $sp, $t0 = the offset,
 * (__stack_top - $t1 + 0x7fff) with its low 16 bits cleared — small
 * numbers only, so o32 and n64 alike. */
#define STACK_TOP jal __emips_stack; nop
#define STACK_CODE move $t1, $sp; LA($sp, __stack_top); beqz $t1, 9f; nop; \
        PTR_SUBU $t0, $sp, $t1; PTR_ADDIU $t0, $t0, 0x7fff; sra $t0, $t0, 16; \
        sll $t0, $t0, 16; PTR_SUBU $sp, $sp, $t0; 9: jr $ra; nop
