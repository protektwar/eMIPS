# MIPS by Example in eMIPS, episode 33 — A guessing game: random numbers
# and dialogs.  MARS's random numbers are java.util.Random streams: 40 seeds
# one, 42 draws from [0, bound).  The seed comes from the time (30): in MARS
# the host's clock, a new game every run; in eMIPS the simulated clock, the
# same game every run — so a check can play it.
        .data
ask:    .asciiz "Guess my number, 1 to 100:"
higher: .asciiz "Higher."
lower:  .asciiz "Lower."
notnum: .asciiz "Not a number."
gotit:  .asciiz "Got it! Guesses: "
itwas:  .asciiz "It was "
again:  .asciiz "Play again?"
logh:   .asciiz " higher\n"
logl:   .asciiz " lower\n"
logok:  .asciiz " right, guesses: "
loggu:  .asciiz "gave up: "
        .text
main:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 0-15 the home area
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        move  $fp, $sp
        li    $v0, 30                 # time: milliseconds in $a0 (low), $a1 (high)
        syscall
        move  $a1, $a0                # the seed: the time's low word
        li    $a0, 0                  # stream 0
        li    $v0, 40                 # set seed
        syscall
game:   jal   play
        li    $v0, 50                 # ConfirmDialog: 0 Yes, 1 No, 2 Cancel
        la    $a0, again
        syscall
        beqz  $a0, game               # Yes: another game
        move  $sp, $fp                # main's epilogue
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        li    $v0, 10
        syscall

# play: one game — a secret 1 to 100, guesses until right (or Cancel)
play:   addiu $sp, $sp, -32           # 28 $ra, 24 $fp, 20 $s0, 16 $s1
        sw    $ra, 28($sp)
        sw    $fp, 24($sp)
        sw    $s0, 20($sp)
        sw    $s1, 16($sp)
        move  $fp, $sp
        li    $v0, 42                 # random int in [0, 100) from stream 0
        li    $a0, 0
        li    $a1, 100
        syscall
        addiu $s0, $a0, 1             # the secret: 1 ... 100, kept in $s0
        li    $s1, 0                  # guesses so far
pask:   li    $v0, 51                 # InputDialogInt
        la    $a0, ask
        syscall
        li    $t0, -2
        beq   $a1, $t0, pquit         # Cancel: give up
        bnez  $a1, pnot               # not a number
        addiu $s1, $s1, 1
        move  $t1, $a0                # the guess
        li    $v0, 1                  # the log: the guess ...
        syscall
        beq   $t1, $s0, pwin
        blt   $t1, $s0, phigh
        li    $v0, 55                 # too big: "Lower."
        la    $a0, lower
        li    $a1, 1
        syscall
        li    $v0, 4
        la    $a0, logl
        syscall
        b     pask
phigh:  li    $v0, 55                 # too small: "Higher."
        la    $a0, higher
        li    $a1, 1
        syscall
        li    $v0, 4
        la    $a0, logh
        syscall
        b     pask
pnot:   li    $v0, 55
        la    $a0, notnum
        li    $a1, 2
        syscall
        b     pask
pwin:   li    $v0, 4                  # " right, guesses: n"
        la    $a0, logok
        syscall
        li    $v0, 1
        move  $a0, $s1
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 56                 # MessageDialogInt
        la    $a0, gotit
        move  $a1, $s1
        syscall
        b     pdone
pquit:  li    $v0, 4                  # "gave up: n"
        la    $a0, loggu
        syscall
        li    $v0, 1
        move  $a0, $s0
        syscall
        li    $v0, 11
        li    $a0, '\n'
        syscall
        li    $v0, 56
        la    $a0, itwas
        move  $a1, $s0
        syscall
pdone:  move  $sp, $fp
        lw    $s1, 16($sp)
        lw    $s0, 20($sp)
        lw    $fp, 24($sp)
        lw    $ra, 28($sp)
        addiu $sp, $sp, 32
        jr    $ra
