# MIPS by Example in eMIPS, episode 30, the challenge solved.  The same
# account, run until it reaches 10000.00: how many years does it take, and
# how much did the fees cost on the way?  month adds each fee to a total
# in memory, so nothing has to pass it back up through the calls.
        .data
fees:   .word 0                       # every fee paid, in cents
reachw: .asciiz "10000.00 reached in year "
colon:  .asciiz ": "
feesw:  .asciiz "fees paid: "
        .text
main:   addiu $sp, $sp, -32           # main's frame: 28 $ra, 24 $fp,
        sw    $ra, 28($sp)            # 20-16 $s0-$s1, 0-15 the home area
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $s0, 100000             # 1000.00 to start
        li    $s1, 0                  # years
mnext:  move  $a0, $s0                # a year at a time ...
        jal   year
        move  $s0, $v0
        addiu $s1, $s1, 1
        blt   $s0, 1000000, mnext     # ... until 10000.00
        li    $v0, 4                  # "10000.00 reached in year y: $amount"
        la    $a0, reachw
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        move  $a0, $s0
        jal   print_money
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 4                  # "fees paid: $amount"
        la    $a0, feesw
        syscall
        lw    $a0, fees
        jal   print_money
        li    $v0, 11
        li    $a0, '\n'
        syscall
        move  $sp, $fp                # main's epilogue
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# year: $a0 = the balance -> $v0 = after 12 months and 5 % interest (it
# calls month: a frame, the balance and the month count in $s0 and $s1)
year:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $s1, 12
ynext:  move  $a0, $s0
        jal   month
        move  $s0, $v0
        addiu $s1, $s1, -1
        bnez  $s1, ynext
        li    $t0, 5
        mul   $t0, $s0, $t0
        li    $t1, 100
        divu  $t0, $t1
        mflo  $t0
        addu  $v0, $s0, $t0
        move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# month: $a0 = the balance -> $v0 = + 100.00 - the fee; the fee also goes
# to fees (it calls fee: a frame, the balance in $s0)
month:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        addiu $s0, $a0, 10000
        move  $a0, $s0
        jal   fee
        lw    $t0, fees               # the total so far, plus this fee
        addu  $t0, $t0, $v0
        sw    $t0, fees
        subu  $v0, $s0, $v0
        move  $sp, $fp
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra

# fee: $a0 = the balance -> $v0 = 1 % of it while under 2000.00, else 0
# (a leaf)
fee:    li    $v0, 0
        bge   $a0, 200000, fdone
        li    $t0, 100
        divu  $a0, $t0
        mflo  $v0
fdone:  jr    $ra

# print_money: $a0 = cents: "$1234.56" (a leaf: system calls only)
print_money:
        li    $t0, 100
        divu  $a0, $t0
        li    $v0, 11
        li    $a0, '$'
        syscall
        li    $v0, 1
        mflo  $a0
        syscall
        li    $v0, 11
        li    $a0, '.'
        syscall
        mfhi  $t1
        li    $t0, 10
        divu  $t1, $t0
        mflo  $a0
        addiu $a0, $a0, '0'
        syscall
        mfhi  $a0
        addiu $a0, $a0, '0'
        syscall
        jr    $ra
