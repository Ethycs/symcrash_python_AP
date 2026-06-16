"""Smoke test for symcrash_workbench session (no OSC, no audio device)."""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import symcrash_workbench.session as wbs
wbs.sd = None  # no audio device in CI-style run

from symcrash_workbench.session import WorkbenchSession

SCRATCH = Path("_smoketest_preset.yaml")
shutil.copy("presets/instruments/dx_marimba_v0.yaml", SCRATCH)

s = WorkbenchSession(SCRATCH, autoplay=False)

# 1. Reads
checks = []
v = s.registry["/amp/decay"].get(s.ir)
checks.append(("amp decay read", abs(v - 500.0) < 1e-6, v))
v = s.registry["/fm/op2/ratio"].get(s.ir)
checks.append(("fm op2 ratio read", abs(v - 2.0) < 1e-3, v))
v = s.registry["/env1/velocity_sens"].get(s.ir)
checks.append(("env1 vel_sens read", abs(v - 0.5) < 1e-6, v))
v = s.registry["/axis/stable/warp"].get(s.ir)
checks.append(("axis stable warp read (empty)", v == 0.0, v))
v = s.registry["/filter/cutoff_hz"].get(s.ir)
checks.append(("filter read on no-Filter preset", v is None, v))

# 2. Render baseline, mutate, render again -> output must change
base = s.render()
assert s.handle("/amp/decay", 1200.0)
assert s.handle("/fm/op2/level", 0.55)
assert s.handle("/axis/stable/warp", 0.4)
assert s.handle("/env1/decay", 300.0)
mod = s.render()
n = min(len(base), len(mod))
diff = int(abs(base[:n].astype("i8") - mod[:n].astype("i8")).max())
checks.append(("mutation audible (max LSB diff)", diff > 100, diff))

# axis knob recruited operators?
tmpl = s.ir.layers.L3_cmmm.emit_templates["T_STABLE"]
checks.append(("warp recruits BandShear", "BandShear" in tmpl.ops_enable,
               tmpl.ops_enable))

# velocity path
s.handle("/velocity", 0.4)
soft = s.render()
n = min(len(mod), len(soft))
vdiff = int(abs(mod[:n].astype("i8") - soft[:n].astype("i8")).max())
checks.append(("velocity changes render", vdiff > 100, vdiff))

# 3. Save -> version bump + values in YAML + comments preserved
assert s.save()
text = SCRATCH.read_bytes().decode("utf-8")
checks.append(("version bumped", "version: 12" in text, None))
checks.append(("decay written", "decay: 1200.0" in text, None))
checks.append(("op level written", "level: 0.55" in text, None))
checks.append(("emphasis written", "warp: 0.4" in text, None))
checks.append(("comments preserved", "faithful Operator recipe" in text, None))
checks.append(("CRLF preserved", "\r\n" in text, None))

# 4. Reload from disk reflects saved state
s.handle("/reload", 1)
time.sleep(0.1)
v = s.registry["/amp/decay"].get(s.ir)
checks.append(("reload sees saved decay", abs(v - 1200.0) < 1e-6, v))
checks.append(("reload sees version", s.ir.meta.version == 12,
               s.ir.meta.version))

# 5. push_state emits sane pairs
pushed = {}
s.push_state(lambda a, v: pushed.setdefault(a, v))
checks.append(("push_state count", len(pushed) > 30, len(pushed)))
checks.append(("push skips absent filter", "/filter/cutoff_hz" not in pushed,
               None))

s.close()
SCRATCH.unlink()

fails = 0
for name, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        fails += 1
    print(f"  {mark}  {name}" + (f"  [{detail}]" if not ok else ""))
print(f"\n{len(checks) - fails}/{len(checks)} passed")
sys.exit(1 if fails else 0)
