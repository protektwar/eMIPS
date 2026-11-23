# Lab 92 — Reading a file: count the lines, words and bytes of
# file_words.txt (as the Unix wc does) and write it in capitals to
# upper.txt.  The file comes with the program (--file, see the last line);
# the program reaches no other file.
# Run:  python3 -m emips run docs/guide/labs/file_words.asm --files-out out
        .data
name:   .asciiz "file_words.txt"
oname:  .asciiz "upper.txt"
nofile: .asciiz "cannot open the file\n"
buf:    .space 64
        .text
main:   li    $v0, 13                 # open file_words.txt for reading
        la    $a0, name
        li    $a1, 0
        syscall
        bltz  $v0, fail
        move  $s0, $v0                # its descriptor
        li    $v0, 13                 # open upper.txt for writing
        la    $a0, oname
        li    $a1, 1
        syscall
        move  $s1, $v0
        li    $s2, 0                  # lines
        li    $s3, 0                  # words
        li    $s4, 0                  # bytes
        li    $s5, 0                  # reads
        li    $s6, 0                  # 1 inside a word
chunk:  li    $v0, 14                 # read up to 64 bytes
        move  $a0, $s0
        la    $a1, buf
        li    $a2, 64
        syscall
        addiu $s5, $s5, 1
        blez  $v0, done               # 0: the end of the file
        move  $t9, $v0                # bytes in this chunk
        addu  $s4, $s4, $v0
        la    $t0, buf
        addu  $t8, $t0, $t9           # the chunk's end
next:   lbu   $t1, 0($t0)
        bne   $t1, '\n', 1f
        addiu $s2, $s2, 1
1:      sltiu $t2, $t1, 33            # space, newline, tab ...: not a word
        beqz  $t2, 2f
        li    $s6, 0
        b     3f
2:      bnez  $s6, 3f                 # the first letter of a word
        addiu $s3, $s3, 1
        li    $s6, 1
3:      sltiu $t2, $t1, 'a'           # a-z -> A-Z
        bnez  $t2, 4f
        sltiu $t2, $t1, 'z' + 1
        beqz  $t2, 4f
        addiu $t1, $t1, -32
        sb    $t1, 0($t0)
4:      addiu $t0, $t0, 1
        bne   $t0, $t8, next
        li    $v0, 15                 # the chunk, in capitals, to upper.txt
        move  $a0, $s1
        la    $a1, buf
        move  $a2, $t9
        syscall
        b     chunk
done:   li    $v0, 16                 # close both
        move  $a0, $s0
        syscall
        move  $a0, $s1
        syscall
        move  $a0, $s2                # lines words bytes reads
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        move  $a0, $s3
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        move  $a0, $s4
        li    $v0, 1
        syscall
        li    $a0, ' '
        li    $v0, 11
        syscall
        move  $a0, $s5
        li    $v0, 1
        syscall
        li    $v0, 10
        syscall
fail:   la    $a0, nofile
        li    $v0, 4
        syscall
        li    $v0, 10
        syscall
# eMIPS: --file file_words.txt
