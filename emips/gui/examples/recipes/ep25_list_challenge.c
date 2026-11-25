/* MIPS by Example in eMIPS, episode 25, the challenge solved (the C version
 * of ep25_list_challenge.asm). */
#include "emips.h"

struct node {
    int value;
    struct node *next;
};

static const int values[4] = {42, 7, 19, 3};

struct node *insert(struct node *head, int v)   /* into a sorted list */
{
    struct node *n = sbrk(sizeof *n), *p;
    n->value = v;
    if (!head || head->value > v) {       /* empty, or smaller than the first */
        n->next = head;
        return n;
    }
    for (p = head; p->next && p->next->value <= v; p = p->next)
        ;
    n->next = p->next;
    p->next = n;
    return head;
}

struct node *remove(struct node *head, int v)   /* the first node with v */
{
    struct node *p;
    if (!head)
        return head;
    if (head->value == v)
        return head->next;
    for (p = head; p->next; p = p->next)
        if (p->next->value == v) {
            p->next = p->next->next;      /* skip over it */
            break;
        }
    return head;
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

    for (int i = 0; i < 4; i++)
        head = insert(head, values[i]);
    print_list("sorted:", head);
    head = remove(head, 19);
    print_list("without 19:", head);
    head = remove(head, 3);
    print_list("without 3:", head);
    return 0;
}
