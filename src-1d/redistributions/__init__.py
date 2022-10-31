REGISTRY = {}

from .ata import ATA
from .rudder import RUDDER

REGISTRY["ata"] = ATA
REGISTRY["rudder"] = RUDDER
REGISTRY["none"] = None

