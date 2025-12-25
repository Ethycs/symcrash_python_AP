import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symcrash_python_ap.safety.unified_renderer import CarrierLiftPipeline
print("Success!")
