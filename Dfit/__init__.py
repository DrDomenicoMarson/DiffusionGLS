from .Dfit import Dcov, XI_CUBIC, BOLTZMANN_K
from .trajectory_reader import TrajectoryReader, NumpyTextReader, get_reader

try:
    from .trajectory_reader import MDAnalysisReader, MDAnalysisMultiReader
except ImportError:
    pass

__all__ = ['Dcov', 'TrajectoryReader', 'NumpyTextReader', 'get_reader']
try:
    __all__.extend(['MDAnalysisReader', 'MDAnalysisMultiReader'])
except NameError:
    pass
