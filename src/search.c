#include <stdlib.h>
#include <stdio.h>
#include <liburing.h>


#define URING_QUEUE_DEPTH 128

/* Singleton uring context for each thread (could have used "option.h" here but black-magic is not worth it) */
typedef struct {
    struct io_uring ring;
    int ready;
} UringContext;


static __thread UringContext thread_uring_context = {0};

// Very basic initialization for now
static struct io_uring* get_thread_uring()
{
    if (!thread_uring_context.ready) {
        if (io_uring_queue_init(URING_QUEUE_DEPTH, &thread_uring_context.ring, 0) < 0) {
            perror("io_uring_queue_init");
            exit(1);
        }
        thread_uring_context.ready = 1;
    }

    return &thread_uring_context.ring;
}

int main(int argc, char** argv)
{
    return 0;
}   