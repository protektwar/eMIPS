# MIPS by Example in eMIPS, episode 36 — A maze game: a file, the display,
# the keyboard.  The maze is a text file (16 lines of 16 characters: #
# wall, . path, S start, E exit), read with MARS's file services; each
# character becomes a pixel of a 16 x 16 display; the w a s d keys typed
# on the MMIO keyboard move the player, q gives up.
        .data
        .eqv  SCREEN, 0x10008000      # the display: 16 x 16 words
        .eqv  KEYBOARD, 0xffff0000    # MMIO: +0 bit 0 = a key is there, +4 the key
        .eqv  ROW, 17                 # bytes a row in the file: 16 cells and '\n'
fname:  .asciiz "ep36_maze.txt"
maze:   .space 272                    # 16 rows of 17 bytes
nofile: .asciiz "cannot read the maze\n"
outin:  .asciiz "out in "
moves:  .asciiz " moves\n"
bye:    .asciiz "gave up\n"
        .text
main:   addiu $sp, $sp, -48           # 44 $ra, 40 $fp, 36-16 $s0-$s5
        sw    $ra, 44($sp)
        sw    $fp, 40($sp)
        sw    $s0, 36($sp)
        sw    $s1, 32($sp)
        sw    $s2, 28($sp)
        sw    $s3, 24($sp)
        sw    $s4, 20($sp)
        sw    $s5, 16($sp)
        move  $fp, $sp
        jal   load                    # the file into maze
        bltz  $v0, fail
        jal   draw                    # the maze on the display; the start
        move  $s0, $v0                # x
        move  $s1, $v1                # y
        li    $s2, 0                  # moves
        move  $a0, $s0
        move  $a1, $s1
        li    $a2, 0x00ffd020         # the player: yellow
        jal   plot
mloop:  jal   getkey
        move  $s3, $s0                # where the key leads: x, y
        move  $s4, $s1
        li    $t0, 'q'
        beq   $v0, $t0, quit
        li    $t0, 'w'
        beq   $v0, $t0, up
        li    $t0, 's'
        beq   $v0, $t0, down
        li    $t0, 'a'
        beq   $v0, $t0, left
        li    $t0, 'd'
        bne   $v0, $t0, mloop         # any other key: nothing
        addiu $s3, $s3, 1             # d: right
        b     try
up:     addiu $s4, $s4, -1
        b     try
down:   addiu $s4, $s4, 1
        b     try
left:   addiu $s3, $s3, -1
try:    move  $a0, $s3                # what is there?
        move  $a1, $s4
        jal   cell
        move  $s5, $v0
        li    $t0, '#'
        beq   $s5, $t0, mloop         # a wall: stay
        move  $a0, $s0                # the old place: path again
        move  $a1, $s1
        li    $a2, 0x00101010
        jal   plot
        move  $s0, $s3                # move
        move  $s1, $s4
        addiu $s2, $s2, 1
        move  $a0, $s0
        move  $a1, $s1
        li    $a2, 0x00ffd020
        jal   plot
        li    $t0, 'E'
        bne   $s5, $t0, mloop
        li    $v0, 4                  # "out in n moves"
        la    $a0, outin
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, moves
        syscall
        b     done
quit:   li    $v0, 4
        la    $a0, bye
        syscall
        b     done
fail:   li    $v0, 4
        la    $a0, nofile
        syscall
done:   move  $sp, $fp                # main's epilogue
        lw    $s5, 16($sp)
        lw    $s4, 20($sp)
        lw    $s3, 24($sp)
        lw    $s2, 28($sp)
        lw    $s1, 32($sp)
        lw    $s0, 36($sp)
        lw    $fp, 40($sp)
        lw    $ra, 44($sp)
        addiu $sp, $sp, 48
        li    $v0, 10
        syscall

# load: the maze file into maze -> $v0 = the bytes read, or -1 (a leaf)
load:   li    $v0, 13                 # open, for reading (0)
        la    $a0, fname
        li    $a1, 0
        li    $a2, 0
        syscall
        bltz  $v0, lfail
        move  $t0, $v0                # the file descriptor
        li    $v0, 14                 # read: 272 bytes at most
        move  $a0, $t0
        la    $a1, maze
        li    $a2, 272
        syscall
        move  $t1, $v0
        li    $v0, 16                 # close
        move  $a0, $t0
        syscall
        move  $v0, $t1
lfail:  jr    $ra

# draw: every cell as a pixel -> $v0, $v1 = the start (S) (a leaf)
draw:   la    $t0, maze               # the next character
        li    $t1, SCREEN             # the next pixel
        li    $t2, 0                  # y
drow:   li    $t3, 0                  # x
dcell:  lbu   $t4, 0($t0)
        li    $t5, 0x00304080         # a wall: blue
        li    $t6, '#'
        beq   $t4, $t6, dput
        li    $t5, 0x0040c040         # the exit: green
        li    $t6, 'E'
        beq   $t4, $t6, dput
        li    $t5, 0x00101010         # the path: black
        li    $t6, 'S'
        bne   $t4, $t6, dput
        move  $v0, $t3                # the start
        move  $v1, $t2
dput:   sw    $t5, 0($t1)
        addiu $t0, $t0, 1
        addiu $t1, $t1, 4
        addiu $t3, $t3, 1
        blt   $t3, 16, dcell
        addiu $t0, $t0, ROW - 16      # skip the row's newline
        addiu $t2, $t2, 1
        blt   $t2, 16, drow
        jr    $ra

# cell(x, y) -> $v0 = the maze's character there (a leaf)
cell:   li    $t0, ROW
        mul   $t0, $a1, $t0           # y * 17 + x
        addu  $t0, $t0, $a0
        lbu   $v0, maze($t0)
        jr    $ra

# plot(x, y, colour): one pixel (a leaf)
plot:   sll   $t0, $a1, 4             # (y * 16 + x) * 4
        addu  $t0, $t0, $a0
        sll   $t0, $t0, 2
        li    $t1, SCREEN
        addu  $t0, $t0, $t1
        sw    $a2, 0($t0)
        jr    $ra

# getkey -> $v0: the next key typed on the MMIO keyboard (a leaf)
getkey: li    $t0, KEYBOARD
gwait:  lw    $t1, 0($t0)             # Receiver Control: bit 0, ready
        andi  $t1, $t1, 1
        beqz  $t1, gwait
        lw    $v0, 4($t0)             # Receiver Data: the key
        jr    $ra
# eMIPS: --display 256x256:16@gp --file ep36_maze.txt
