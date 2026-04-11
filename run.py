#!/usr/bin/env python3

import os
import sys


def _register_tensorrt_dlls():
    """Make pip-installed TensorRT DLLs loadable before ORT imports.

    The ``tensorrt`` / ``tensorrt_cuNN`` wheels drop ``nvinfer_*.dll`` into
    ``site-packages/tensorrt_libs``, which is NOT on PATH by default. Without
    this step, ORT's TensorrtExecutionProvider fails to load even though
    ``pip show tensorrt`` reports it installed. We scan every site-packages
    dir for a ``tensorrt_libs`` folder and register each one.

    Must run before anything imports ``onnxruntime`` (which happens the moment
    we touch ``roop.core``).
    """
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return

    try:
        import site
        search_roots = list(site.getsitepackages())
    except Exception:
        search_roots = []
    search_roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))

    seen = set()
    for root in search_roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        candidate = os.path.join(root, "tensorrt_libs")
        if os.path.isdir(candidate):
            try:
                os.add_dll_directory(candidate)
            except OSError:
                pass


_register_tensorrt_dlls()

from roop import core  # noqa: E402

if __name__ == "__main__":
    core.run()
