import time


def log_time(get_time=False, printer=print, method_name=None):
    def _log_time(method):
        def timed(*args, **kw):
            ts = time.perf_counter()
            result = method(*args, **kw)
            te = time.perf_counter()
            walltime = te - ts
            # Can not define a local variable with the same name as one of the outer function params so we need to change the name from method_name to methodname
            # Ref: https://stackoverflow.com/a/28739003
            methodname = method_name
            if methodname is None:
                if hasattr(method, "__qualname__"):
                    methodname = method.__qualname__
                elif hasattr(method, "name"):
                    methodname = method.name
                elif hasattr(method, "__name__"):
                    methodname = method.__name__
                else:
                    methodname = "unknown_fn"
            printer(f"{methodname} runtime: {(te - ts):.3f}s")
            if get_time:
                return result, walltime
            else:
                return result

        return timed

    return _log_time
