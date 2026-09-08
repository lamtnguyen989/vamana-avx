// Pretty much a crack-pot macro magic for Option<T> in C

#ifndef OPTION_H
#define OPTION_H

#define DEFINE_OPTION(T)    \
    typedef struct {        \
        int has_value;      \
        T value;            \
    } Option_##T;

#define OPTION(T) Option_##T
#define OPTION_NONE(T)      ((Option_##T){ .has_value = 0 })
#define OPTION_SOME(T, v)   ((Option_##T){ .has_value = 1, .value = (v) })
#define OPTION_IS_SOME(opt) ((opt).has_value)
#define OPTION_IS_NONE(opt) (!(opt).has_value)
#define OPTION_UNWRAP(opt)  ((opt).value)
#define OPTION_UNWRAP_OR(opt, default_val) ((opt).has_value ? (opt).value : (default_val))


#endif /* OPTION_H */