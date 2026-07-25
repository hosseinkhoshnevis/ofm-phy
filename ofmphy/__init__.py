"""ofmphy: baseband PHY simulation of an 800G-class coherent optical
fibre modem: DP-16QAM at the 800ZR symbol rate with the OpenROADM oFEC.
"""

from .config import SimConfig
from .ofec import OfecCodec
from .sim import run_link

__version__ = "0.8.0"
__all__ = ["SimConfig", "OfecCodec", "run_link", "__version__"]
