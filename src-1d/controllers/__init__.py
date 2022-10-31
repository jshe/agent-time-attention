REGISTRY = {}

from .basic_controller import BasicMAC
from .distributed_controller import DistributedMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["distributed_mac"] = DistributedMAC