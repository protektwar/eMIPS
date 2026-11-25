/* MIPS by Example in eMIPS, episode 25 — A linked list with sbrk (the C
 * version of ep25_list.asm). */
#include "emips.h"

struct node {
    int value;
    struct node *next;                    /* 0 (null) after the last */
};

struct node *push(struct node *head, int v)
{
    struct node *n = sbrk(sizeof *n);     /* 8 bytes from the heap */
    n->value = v;
    n->next = head;
    return n;
}

void append(struct node *head, int v)     /* head: not empty */
{
    struct node *n = sbrk(sizeof *n);
    while (head->next)
        head = head->next;
    n->value = v;
    n->next = 0;
    head->next = n;
}

int length(const struct node *p)
{
    int n = 0;
    for (; p; p = p->next)
        n++;
    return n;
}

struct node *reverse(struct node *p)
{
    struct node *prev = 0;
    while (p) {
        struct node *next = p->next;      /* save next first ... */
        p->next = prev;                   /* ... then point back */
        prev = p;
        p = next;
    }
    return prev;
}

static void print_list(const char *text, const struct node *p)
{
    print_string(text);
    for (; p; p = p->next) {
        print_char(' ');
        print_int(p->value);
    }
    print_char('\n');
}

int main(void)
{
    struct node *head = 0;

    head = push(head, 3);
    head = push(head, 7);
    head = push(head, 11);
    append(head, 20);
    print_list("list:", head);
    print_string("length ");
    print_int(length(head));
    print_char('\n');
    head = reverse(head);
    print_list("reversed:", head);
    return 0;
}
