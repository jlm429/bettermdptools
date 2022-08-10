import functools
import time


def print_runtime(func):
    @functools.wraps(func)
    def wrapper_print_runtime(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        running_time = end - start
        print("runtime = %.2f seconds" % running_time)
        return value
    return wrapper_print_runtime


# from https://wiki.python.org/moin/PythonDecoratorLibrary
def add_to(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        setattr(func, args[0].__name__, args[0])
        return func
    return decorator


# from https://realpython.com/
def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug
