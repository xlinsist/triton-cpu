import time
from functools import wraps

import numpy as np
import triton


def _get_cpu_driver():
    candidate_paths = [
        ("triton.backends.cpu.driver", "CPUDriver"),
        ("triton.backends.triton_cpu.driver", "CPUDriver"),
    ]

    for module_name, class_name in candidate_paths:
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


def select_cpu_backend():
    cpu_driver_cls = _get_cpu_driver()
    if cpu_driver_cls is None:
        raise RuntimeError(
            "Cannot find a CPUDriver in this Triton installation. "
            "Tried known paths for triton-cpu / triton-shared."
        )

    try:
        triton.runtime.driver.set_active(cpu_driver_cls())
        return
    except Exception:
        pass

    try:
        triton.runtime.driver.active = cpu_driver_cls()
        return
    except Exception as e:
        raise RuntimeError(
            "Found CPUDriver, but failed to activate it through Triton runtime API."
        ) from e


def _maybe_call_sync():
    candidates = [
        getattr(triton, "synchronize", None),
        getattr(getattr(triton, "runtime", None), "synchronize", None),
    ]

    driver_obj = getattr(getattr(triton, "runtime", None), "driver", None)
    if driver_obj is not None:
        candidates.append(getattr(driver_obj, "synchronize", None))

    for fn in candidates:
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def measure(
    repeats=20,
    warmup=5,
    percentiles=(50, 90, 99),
    timers=None,
    sync=False,
    return_stats=False,
):
    if timers is None:
        timers = {
            "Wall": time.perf_counter,
            "CPU": time.process_time,
        }

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(
                f"{func.__name__}{args} {kwargs}, "
                f"warmup={warmup}, repeats={repeats}, results in seconds"
            )

            result = None
            for _ in range(warmup):
                result = func(*args, **kwargs)
                if sync:
                    _maybe_call_sync()

            times = {name: [] for name in timers}
            for _ in range(repeats):
                starts = {name: timer() for name, timer in timers.items()}

                result = func(*args, **kwargs)

                if sync:
                    _maybe_call_sync()

                for name, timer in timers.items():
                    times[name].append(timer() - starts[name])

            stats = {}
            for name, values in times.items():
                arr = np.asarray(values, dtype=np.float64)
                timer_stats = {
                    "avg": float(np.mean(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "std": float(np.std(arr)),
                }

                if percentiles:
                    pct_values = np.percentile(arr, percentiles)
                    timer_stats["percentiles"] = {
                        float(p): float(v) for p, v in zip(percentiles, pct_values)
                    }
                else:
                    timer_stats["percentiles"] = {}

                stats[name] = timer_stats

                print(
                    f"{name}: "
                    f"Avg={timer_stats['avg']:.6f}, "
                    f"Min={timer_stats['min']:.6f}, "
                    f"Std={timer_stats['std']:.6f}, ",
                    end="",
                )
                for p, v in timer_stats["percentiles"].items():
                    print(f"{p:g}pp={v:.6f}, ", end="")
                print(f"Max={timer_stats['max']:.6f}")

            if return_stats:
                return result, stats
            return result

        return wrapper

    return decorator
