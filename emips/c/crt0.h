/* crt0.h — crt0.S's macros (python3 -m emips cc), for o32 (MIPS32) and
 * n64 (MIPS64).  (In a header so that crt0.S's lines stay where the
 * prebuilt executables' line tables say they are.) */
#if _MIPS_SZPTR == 64
#define PTR_ADDIU daddiu        /* n64: 64-bit pointers */
#else
#define PTR_ADDIU addiu
#endif

/* An address: lui + addiu, the 32-bit sum sign-extended.  (n64's la with
 * -msym32 is lui + daddiu: for __stack_top = 0x7FFFEFF0, %hi = 0x8000 and
 * the 64-bit sum is 0xFFFFFFFF7FFFEFF0.) */
#define LA(reg, sym) lui reg, %hi(sym); addiu reg, reg, %lo(sym)
