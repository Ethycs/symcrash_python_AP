import sounddevice as sd

# Check device 20 (WASAPI Speakers)
d = sd.query_devices(20)
print("Device 20:")
print(d)
print()
print("hostapi:", sd.query_hostapis(d['hostapi']))
