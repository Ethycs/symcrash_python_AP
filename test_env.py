from symcrash_python_ap.envelope_scheduler import EnvelopeScheduler, ADSRSpec

env = EnvelopeScheduler(ADSRSpec(attack_ms=10.0, decay_ms=42.0, sustain=0.75, release_ms=85.0))
env.trigger_note()

for i in range(5):
    gain = env.tick()
    print(f'Block {i}: gain={gain}')
