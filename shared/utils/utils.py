import inspect
import warnings

def get_valid_kwargs(func, argdict):
    # https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives
    arguments = inspect.signature(func).parameters

    if any([p.kind == p.VAR_KEYWORD for p in arguments.values()]):
        # Accepts variable keyword arguments
        return argdict

    return {key: val for key, val in argdict.items() if key in set(arguments)}


def call_valid_kwargs(func, kwargs_dict, args_tuple=tuple()):
    kwargs = get_valid_kwargs(func, kwargs_dict)
    if extra_kwargs := sorted(set(kwargs_dict) - set(kwargs)):
        warnings.warn(f"Function {func} ignored extra arguments {extra_kwargs}")
    return func(*args_tuple, **kwargs)
