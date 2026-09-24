"""FlashRT — Framework-agnostic CUDA Graph capture/replay.

Uses CUDA Runtime API directly via ctypes. Works with any framework
(PyTorch, JAX, or raw CUDA) because it operates at the stream level.

Usage:
    graph = CUDAGraph()
    stream = graph.create_stream()

    # Warmup
    my_kernel(args..., stream)

    # Capture
    graph.begin_capture(stream)
    my_kernel(args..., stream)
    graph.end_capture()

    # Replay (zero dispatch overhead)
    graph.replay(stream)
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

# Load CUDA runtime
_cudart = ctypes.CDLL("libcudart.so")

# Declare exact prototypes for every entry point we call (ctypes otherwise
# passes Python ints as 32-bit c_int; cudaGraphInstantiate*'s flags is a
# 64-bit ``unsigned long long``).
#
# Instantiate via ``cudaGraphInstantiateWithFlags``: the legacy
# ``cudaGraphInstantiate`` symbol segfaults INSIDE the tegra driver on
# L4T r35.6 iGPU for ANY graph — even a single-node memset (reproduced
# 2026-09-24, and independent of which libcudart is loaded). torch's native
# capture works on the same box precisely because it calls the WithFlags
# entry. Same choice the AMD twin made (hipGraphInstantiateWithFlags).
for _fn in (_cudart.cudaStreamCreate,
            _cudart.cudaStreamBeginCapture,
            _cudart.cudaStreamEndCapture,
            _cudart.cudaGraphLaunch,
            _cudart.cudaStreamSynchronize):
    _fn.restype = ctypes.c_int  # cudaError_t

_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
_cudart.cudaStreamEndCapture.argtypes = [ctypes.c_void_p,
                                         ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]

# cudart < 11.4 has no WithFlags entry; fall back to the legacy symbol there.
if hasattr(_cudart, "cudaGraphInstantiateWithFlags"):
    _instantiate = _cudart.cudaGraphInstantiateWithFlags
    _instantiate.restype = ctypes.c_int
    _instantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_void_p,
                             ctypes.c_uint64]
    _instantiate_name = "cudaGraphInstantiateWithFlags"
else:
    _instantiate = _cudart.cudaGraphInstantiate
    _instantiate.restype = ctypes.c_int
    _instantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                             ctypes.c_void_p,
                             ctypes.c_uint64]
    _instantiate_name = "cudaGraphInstantiate"


def _check(status, msg=""):
    if status != 0:
        raise RuntimeError(f"CUDA error {status}: {msg}")


class CUDAGraph:
    """Framework-agnostic CUDA Graph using raw CUDA Runtime API."""

    def __init__(self):
        self._graph = ctypes.c_void_p()
        self._graph_exec = ctypes.c_void_p()
        self._captured = False

    def create_stream(self) -> ctypes.c_void_p:
        """Create a new CUDA stream for capture."""
        stream = ctypes.c_void_p()
        _check(_cudart.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def begin_capture(self, stream: ctypes.c_void_p):
        """Begin CUDA Graph capture on the given stream.

        All CUDA operations on this stream after this call will be
        recorded into the graph instead of being executed.
        """
        # cudaStreamCaptureModeRelaxed=2: only capture ops on THIS stream.
        # Global mode (0) blocks ALL streams, conflicting with XLA's background ops.
        _check(_cudart.cudaStreamBeginCapture(stream, 2), "cudaStreamBeginCapture")

    def end_capture(self, stream: ctypes.c_void_p):
        """End capture and instantiate the graph for replay."""
        _check(_cudart.cudaStreamEndCapture(stream, ctypes.byref(self._graph)),
               "cudaStreamEndCapture")
        _check(_instantiate(
            ctypes.byref(self._graph_exec), self._graph, 0),
               _instantiate_name)
        self._captured = True

    def replay(self, stream: ctypes.c_void_p):
        """Replay the captured graph (single CPU instruction → GPU replay)."""
        if not self._captured:
            raise RuntimeError("No graph captured")
        _check(_cudart.cudaGraphLaunch(self._graph_exec, stream), "cudaGraphLaunch")

    def sync(self, stream: ctypes.c_void_p):
        """Synchronize stream."""
        _check(_cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    @property
    def captured(self) -> bool:
        return self._captured
