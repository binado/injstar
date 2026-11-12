"""Process-wide JAX and CPU thread configuration for entry points.

Call :func:`configure_runtime` before importing libraries that initialize CPU
thread pools and before JAX initializes its backend.

Ported from ``astrogwb.runtime`` (astrogwb 0.7.0).
"""

import os

CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def set_cpu_threads(cpu_threads: int) -> None:
    """Limit the CPU thread pools of libraries imported after this call.

    Parameters
    ----------
    cpu_threads
        Positive thread limit.
    """
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    for name in CPU_THREAD_ENV_VARS:
        os.environ[name] = str(cpu_threads)


def configure_runtime(
    *, platform: str = "auto", cpu_threads: int | None = None
) -> None:
    """Configure the JAX platform, CPU thread limits, and 64-bit precision.

    Parameters
    ----------
    platform
        One of ``auto``, ``cpu``, ``cuda``, or ``tpu``. ``auto`` preserves an
        existing ``JAX_PLATFORMS`` setting.
    cpu_threads
        Positive thread limit, or ``None`` to leave existing settings unchanged.
    """
    if platform not in {"auto", "cpu", "cuda", "tpu"}:
        raise ValueError(f"unknown platform {platform!r}")

    if cpu_threads is not None:
        set_cpu_threads(cpu_threads)
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={cpu_threads}"
        )
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform

    import jax

    jax.config.update("jax_enable_x64", True)
