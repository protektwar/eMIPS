# MIPS by Example in eMIPS, episode 36's challenge, solved: a robot walks
# the maze by the right-hand rule — keep a hand on the wall to your right:
# turn right if you can, else go straight, else left, else back.  No keys;
# the cells it walked stay marked (a trail), so its whole route is seen.
        .data
        .eqv  SCREEN, 0x10008000      # the display: 16 x 16 words
        .eqv  ROW, 17                 # bytes a row in the file: 16 cells and '\n'
fname:  .asciiz "ep36_maze.txt"
maze:   .space 272                    # 16 rows of 17 bytes
nofile: .asciiz "cannot read the maze\n"
outin:  .asciiz "robot out in "
moves:  .asciiz " moves\n"
dx:     .word 0, 1, 0, -1           # north, east, south, west
dy:     .word -1, 0, 1, 0
turns:  .word 1, 0, 3, 2             # right, straight, left, back
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
        li    $s5, 1                  # facing: east (0 N, 1 E, 2 S, 3 W)
        move  $a0, $s0
        move  $a1, $s1
        li    $a2, 0x00ffd020         # the player: yellow
        jal   plot
mloop:  li    $s4, 0                  # the turns to try: right, straight, left, back
tturn:  sll   $t0, $s4, 2
        lw    $t0, turns($t0)
        addu  $s3, $s5, $t0           # the direction to try: (d + turn) mod 4
        andi  $s3, $s3, 3
        sll   $t0, $s3, 2
        lw    $t1, dx($t0)
        lw    $t2, dy($t0)
        addu  $a0, $s0, $t1           # what is there?
        addu  $a1, $s1, $t2
        jal   cell
        li    $t0, '#'
        bne   $v0, $t0, tgo
        addiu $s4, $s4, 1             # a wall: the next turn
        b     tturn
tgo:    move  $s4, $v0                # the cell's character
        move  $a0, $s0                # the old place: the trail
        move  $a1, $s1
        li    $a2, 0x00806010
        jal   plot
        move  $s5, $s3                # face that way and step
        sll   $t0, $s5, 2
        lw    $t1, dx($t0)
        lw    $t2, dy($t0)
        addu  $s0, $s0, $t1
        addu  $s1, $s1, $t2
        addiu $s2, $s2, 1
        move  $a0, $s0
        move  $a1, $s1
        li    $a2, 0x00ffd020
        jal   plot
        li    $t0, 'E'
        bne   $s4, $t0, mloop
        li    $v0, 4                  # "robot out in n moves"
        la    $a0, outin
        syscall
        li    $v0, 1
        move  $a0, $s2
        syscall
        li    $v0, 4
        la    $a0, moves
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

# eMIPS: --display 256x256:16@gp --file ep36_maze.txt
