# MIPS by Example in eMIPS, episode 30 — A simulation with nested calls: a
# savings account, in cents.  Every month 100.00 goes in and a fee of 1 %
# comes out while the balance is under 2000.00; every year 5 % interest is
# added.  main calls simulate, which calls year, which calls month, which
# calls fee: four frames on the stack at once, each with its own $ra.
        .data
yearw:  .asciiz "year "
colon:  .asciiz ": "
        .text
main:   addiu $sp, $sp, -24           # main's frame: 20 $ra, 16 $fp,
        sw    $ra, 20($sp)            # 0-15 the home area
        sw    $fp, 16($sp)
        move  $fp, $sp
        li    $a0, 100000             # 1000.00 to start
        li    $a1, 3                  # for 3 years
        jal   simulate
        move  $sp, $fp                # main's epilogue
        lw    $fp, 16($sp)
        lw    $ra, 20($sp)
        addiu $sp, $sp, 24
        li    $v0, 10
        syscall

# simulate: $a0 = the balance, $a1 = years: print the balance after each
# year (it calls year and print_money: a frame, the balance, the years and
# the year number in $s0-$s2)
simulate:
        addiu $sp, $sp, -40           # 36 $ra, 32 $fp, 28-20 $s0-$s2
        sw    $ra, 36($sp)
        sw    $fp, 32($sp)
        sw    $s0, 28($sp)
        sw    $s1, 24($sp)
        sw    $s2, 20($sp)
        move  $fp, $sp
        move  $s0, $a0
        move  $s1, $a1
        li    $s2, 1
snext:  move  $a0, $s0                # a year
        jal   year
        move  $s0, $v0
        li    $v0, 4                  # "year y: $amount"
        la    $a0, yearw
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, colon
        syscall
        move  $a0, $s0
        jal   print_money
        li    $v0, 11
        li    $a0, '\n'
        syscall
        addiu $s2, $s2, 1
        ble   $s2, $s1, snext
        move  $sp, $fp
        lw    $s2, 20($sp)
        lw    $s1, 24($sp)
        lw    $s0, 28($sp)
        lw    $fp, 32($sp)
        lw    $ra, 36($sp)
        addiu $sp, $sp, 40
        jr    $ra

# year: $a0 = the balance -> $v0 = after 12 months and the interest (it
# calls month: a frame, the balance and the month count in $s0 and $s1)
year:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        move  $s0, $a0
        li    $s1, 12
ynext:  move  $a0, $s0                # a month
        jal   month
        move  $s0, $v0
        addiu $s1, $s1, -1
        bnez  $s1, ynext
        li    $t0, 5                  # + 5 % interest: balance × 5 / 100
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

# month: $a0 = the balance -> $v0 = + the deposit - the fee (it calls fee:
# a frame for its $ra; the balance in $s0)
month:  addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0
        sw    $ra, 28($sp)            # fee's jal will overwrite $ra
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        move  $fp, $sp
        addiu $s0, $a0, 10000         # the deposit: 100.00
        move  $a0, $s0
        jal   fee
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
        li    $v0, 1                  # the whole part
        mflo  $a0
        syscall
        li    $v0, 11
        li    $a0, '.'
        syscall
        mfhi  $t1                     # the cents: two digits
        li    $t0, 10
        divu  $t1, $t0
        mflo  $a0
        addiu $a0, $a0, '0'
        syscall
        mfhi  $a0
        addiu $a0, $a0, '0'
        syscall
        jr    $ra
