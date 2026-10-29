"""Array implementations of the math kernels.

This package currently delegates to :mod:`matlib._list` until the
Buffer-native kernels are implemented.
"""

supported = ["Array"]

from . import broadcasting as _broadcasting
from . import linalg as _linalg
from . import manipulation as _manipulation
from . import ops as _ops
from . import stats as _stats
from .broadcasting import *
from .linalg import *
from .manipulation import *
from .ops import *
from .stats import *
from .ops import _apply_unary

__all__ = [
    *_broadcasting.__all__,
    *_linalg.__all__,
    *_manipulation.__all__,
    *_ops.__all__,
    *_stats.__all__,
]
