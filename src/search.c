#include <stdio.h>
#include <liburing.h>


#define URING_QUEUE_DEPTH 128

typedef struct {
    struct io_uring ring;
    int ready;
} UringContext;

int main(int argc, char** argv)
{
    return 0;
}   