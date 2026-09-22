import time
from typing import Tuple, Any, Callable

def measure_inference_time_ms(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Executes a callable function and measures wall-clock execution time in milliseconds.
    
    Returns:
        Tuple containing (function_result, elapsed_time_in_ms)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0
    return result, round(elapsed_ms, 4)