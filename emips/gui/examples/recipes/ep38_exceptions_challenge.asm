# MIPS by Example in eMIPS, episode 38 — the challenge, solved: the handler
# also catches a store to a bad address (5 = AdES) and a break (9 = Bp), and
# counts each kind; main prints the counts at the end.  (The episode: an
# overflow, a division by zero, a bad address.)  When an instruction fails,
# the processor saves its address in EPC, the reason in Cause (bits 6..2:
# ExcCode) and, for an address, the address in BadVAddr, sets Status.EXL and
# jumps to 0x80000180: the general exception vector, where .ktext puts this
# program's handler.  The handler says what happened and returns past the
# instruction that failed (EPC + 4) with ERET.
        .data
plus:   .asciiz " + "
divide: .asciiz " / "
equals: .asciiz " = "
skip:   .asciiz "skipped\n"
wordat: .asciiz "word at "
storeat: .asciiz "store at "
brk:    .asciiz "break = "
total:  .asciiz "exceptions by kind:"
space:  .asciiz " "

        .kdata                        # the handler's own data (kernel)
saved:  .word 0, 0, 0                 # $v0, $a0, $t0 of the interrupted code
failed: .word 0                       # 1: the last try failed (main reads it)
count:  .word 0                       # exceptions handled
kinds:  .space 128                    # kinds[ExcCode]: how many of each (32 codes)
head:   .asciiz "[exception "
colon:  .asciiz ": "
tail:   .asciiz "] "
nstore: .asciiz "bad store "
nbreak: .asciiz "breakpoint"
nover:  .asciiz "overflow"
ntrap:  .asciiz "trap (divide by zero)"
naddr:  .asciiz "bad address "
nother: .asciiz "unexpected"

        .text
main:   addiu $sp, $sp, -24           # 20 $ra, 16 $fp, 0-15 the home area
        sw    $ra, 20($sp)
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $a0, 7                  # 7 + 5: fine
        li    $a1, 5
        jal   try_add
        li    $a0, 0x7fffffff         # the largest int + 1: overflow
        li    $a1, 1
        jal   try_add
        li    $a0, 100                # 100 / 7: fine
        li    $a1, 7
        jal   try_div
        li    $a0, 42                 # 42 / 0: the trap
        li    $a1, 0
        jal   try_div
        li    $a0, 0x10010001         # a word at an odd address: bad address
        jal   try_load
        li    $a0, 0x10010002         # a store there: bad address (store)
        jal   try_store
        jal   try_break               # a break instruction
        jal   summary                 # how many of each kind
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# try_add(a, b): "a + b = sum", or "skipped" when the add failed (a leaf)
try_add:
        move  $t0, $a0
        move  $t1, $a1
        li    $v0, 1                  # a
        syscall
        li    $v0, 4                  # " + "
        la    $a0, plus
        syscall
        li    $v0, 1                  # b
        move  $a0, $t1
        syscall
        li    $v0, 4                  # " = "
        la    $a0, equals
        syscall
        sw    $zero, failed           # not failed (yet)
        add   $a0, $t0, $t1           # add traps on a signed overflow
        b     result

# try_div(a, b): "a / b = quotient", or "skipped" (a leaf)
try_div:
        move  $t0, $a0
        move  $t1, $a1
        li    $v0, 1
        syscall
        li    $v0, 4                  # " / "
        la    $a0, divide
        syscall
        li    $v0, 1
        move  $a0, $t1
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        sw    $zero, failed
        teq   $t1, $zero, 7           # b = 0: trap (code 7), as GCC writes it
        div   $zero, $t0, $t1
        mflo  $a0
        b     result

# try_store(address): "store at address = 7", or "skipped" (a leaf)
try_store:
        move  $t0, $a0
        li    $v0, 4                  # "store at "
        la    $a0, storeat
        syscall
        li    $v0, 34
        move  $a0, $t0
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        sw    $zero, failed
        li    $a0, 7
        sw    $a0, 0($t0)             # an odd address: Address Error (store)
        b     result

# try_break(): "break = 0", or "skipped" (a leaf)
try_break:
        li    $v0, 4                  # "break = "
        la    $a0, brk
        syscall
        sw    $zero, failed
        li    $a0, 0
        break 3                       # Bp: a breakpoint (code 3)
        b     result

# summary(): "exceptions by kind: 4=1 5=1 ..." for every ExcCode seen
# (a leaf)
summary:
        li    $v0, 4
        la    $a0, total
        syscall
        li    $t0, 0                  # the code
        la    $t1, kinds
sloop:  lw    $t2, 0($t1)
        beqz  $t2, snext
        li    $v0, 4                  # " code=count"
        la    $a0, space
        syscall
        li    $v0, 1
        move  $a0, $t0
        syscall
        li    $v0, 11
        li    $a0, '='
        syscall
        li    $v0, 1
        move  $a0, $t2
        syscall
snext:  addiu $t0, $t0, 1
        addiu $t1, $t1, 4
        blt   $t0, 32, sloop
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra

# try_load(address): "word at address = the word", or "skipped" (a leaf)
try_load:
        move  $t0, $a0
        li    $v0, 4                  # "word at "
        la    $a0, wordat
        syscall
        li    $v0, 34                 # the address in hex
        move  $a0, $t0
        syscall
        li    $v0, 4
        la    $a0, equals
        syscall
        sw    $zero, failed
        lw    $a0, 0($t0)             # an odd address: Address Error (load)

result: lw    $t2, failed             # (shared by the three: $a0 = the value)
        bnez  $t2, skipped
        li    $v0, 1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        jr    $ra
skipped:
        li    $v0, 4
        la    $a0, skip
        syscall
        jr    $ra

# the exception handler: the processor comes here with Status.EXL = 1
# (interrupts off), EPC = the failed instruction, Cause = why.  It may only
# change $k0 / $k1: everything else it uses is saved first and restored.
        .ktext 0x80000180
        .set  noat
        move  $k1, $at                # $at first: la / sw below use it
        .set  at
        sw    $v0, saved
        sw    $a0, saved+4
        sw    $t0, saved+8
        li    $v0, 4                  # "[exception "
        la    $a0, head
        syscall
        mfc0  $k0, $13                # Cause
        srl   $k0, $k0, 2
        andi  $k0, $k0, 31            # ExcCode, bits 6..2
        li    $v0, 1
        move  $a0, $k0
        syscall
        li    $v0, 4                  # ": "
        la    $a0, colon
        syscall
        li    $t0, 12                 # 12 = Ov: arithmetic overflow
        la    $a0, nover
        beq   $k0, $t0, hname
        li    $t0, 13                 # 13 = Tr: a trap (teq ...)
        la    $a0, ntrap
        beq   $k0, $t0, hname
        li    $t0, 9                  # 9 = Bp: a break instruction
        la    $a0, nbreak
        beq   $k0, $t0, hname
        li    $t0, 5                  # 5 = AdES: a bad address (store)
        la    $a0, nstore
        beq   $k0, $t0, haddr
        li    $t0, 4                  # 4 = AdEL: a bad address (load)
        la    $a0, nother
        bne   $k0, $t0, hname
        la    $a0, naddr
haddr:  li    $v0, 4                  # "bad address " and BadVAddr
        syscall
        li    $v0, 34
        mfc0  $a0, $8                 # BadVAddr
        syscall
        b     hdone
hname:  li    $v0, 4
        syscall
hdone:  li    $v0, 4                  # "] "
        la    $a0, tail
        syscall
        li    $t0, 1                  # tell the program: it failed
        sw    $t0, failed
        lw    $t0, count              # count it
        addiu $t0, $t0, 1
        sw    $t0, count
        sll   $k0, $k0, 2             # and its kind: kinds[ExcCode] + 1
        lw    $t0, kinds($k0)
        addiu $t0, $t0, 1
        sw    $t0, kinds($k0)
        mfc0  $k0, $14                # EPC: the failed instruction ...
        addiu $k0, $k0, 4             # ... skipped
        mtc0  $k0, $14
        lw    $t0, saved+8            # everything back as it was
        lw    $a0, saved+4
        lw    $v0, saved
        .set  noat
        move  $at, $k1
        .set  at
        ehb                           # (the new EPC seen by ERET)
        eret                          # back to EPC, Status.EXL = 0
