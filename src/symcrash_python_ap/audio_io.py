"""
Audio I/O Module
================

Centralized audio file I/O for the symcrash pipeline.
All audio output should go through this module.

Features:
- Automatic timestamping of output files
- Configurable output directory
- Support for multiple formats (WAV, etc.)
- Consistent stereo/mono handling
"""

import os
import wave
import struct
import numpy as np
from datetime import datetime
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from .primitives import Q15_NEG_ONE, Q15_ONE, Q15_SCALE


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BIT_DEPTH = 16
DEFAULT_OUTPUT_DIR = "audio_out"


@dataclass
class AudioMetadata:
    """Metadata about an audio file."""
    filepath: str
    filename: str
    format: str
    sample_rate: int
    channels: int
    bit_depth: int
    duration_sec: float
    n_samples: int
    peak_amplitude: float
    rms_amplitude: float
    timestamp: str


# =============================================================================
# AUDIO WRITER CLASS
# =============================================================================

class AudioWriter:
    """
    Centralized audio file writer with automatic timestamping.

    Usage:
        writer = AudioWriter()
        writer.write_audio("my_sound", L, R)
        # Creates: audio_out/my_sound-20251219-114532.wav

        # Or without timestamp:
        writer.write_audio("my_sound", L, R, timestamp=False)
        # Creates: audio_out/my_sound.wav
    """

    def __init__(
        self,
        output_dir: str = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bit_depth: int = DEFAULT_BIT_DEPTH,
        timestamp_format: str = "%Y%m%d-%H%M%S",
    ):
        """
        Initialize the audio writer.

        output_dir: Directory for output files (default: audio_out relative to project root)
        sample_rate: Sample rate in Hz (default: 48000)
        bit_depth: Bit depth (default: 16)
        timestamp_format: strftime format for timestamps
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.timestamp_format = timestamp_format

        # Determine output directory
        if output_dir is None:
            # Find project root (go up from src/symcrash_python_ap)
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(module_dir))
            self.output_dir = os.path.join(project_root, DEFAULT_OUTPUT_DIR)
        else:
            self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Generate a timestamp string."""
        return datetime.now().strftime(self.timestamp_format)

    def _build_filepath(
        self,
        filename: str,
        format: str = "wav",
        timestamp: bool = True,
        subdirectory: str = None,
    ) -> str:
        """Build the full output filepath."""
        # Clean filename (remove extension if provided)
        base_name = os.path.splitext(filename)[0]

        # Add timestamp if requested
        if timestamp:
            base_name = f"{base_name}-{self._get_timestamp()}"

        # Add extension
        full_name = f"{base_name}.{format}"

        # Build path
        if subdirectory:
            output_path = os.path.join(self.output_dir, subdirectory)
            os.makedirs(output_path, exist_ok=True)
        else:
            output_path = self.output_dir

        return os.path.join(output_path, full_name)

    def _normalize_audio(
        self,
        L: Union[List[int], np.ndarray],
        R: Union[List[int], np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Normalize audio data to int16 arrays.

        Returns: (L_array, R_array, n_channels)
        """
        # Convert to numpy arrays
        L_arr = np.array(L, dtype=np.float64)

        if R is None:
            # Mono
            R_arr = L_arr.copy()
            n_channels = 1
        else:
            R_arr = np.array(R, dtype=np.float64)
            n_channels = 2

        # Clip to int16 range
        L_arr = np.clip(L_arr, Q15_NEG_ONE, Q15_ONE).astype(np.int16)
        R_arr = np.clip(R_arr, Q15_NEG_ONE, Q15_ONE).astype(np.int16)

        return L_arr, R_arr, n_channels

    def _compute_metadata(
        self,
        filepath: str,
        L: np.ndarray,
        R: np.ndarray,
        n_channels: int,
        format: str,
    ) -> AudioMetadata:
        """Compute metadata for the audio file."""
        L_float = L.astype(np.float64)
        R_float = R.astype(np.float64)

        # Combined for stereo analysis
        if n_channels == 2:
            combined = np.concatenate([L_float, R_float])
        else:
            combined = L_float

        peak = np.max(np.abs(combined))
        rms = np.sqrt(np.mean(combined ** 2))

        return AudioMetadata(
            filepath=filepath,
            filename=os.path.basename(filepath),
            format=format,
            sample_rate=self.sample_rate,
            channels=n_channels,
            bit_depth=self.bit_depth,
            duration_sec=len(L) / self.sample_rate,
            n_samples=len(L),
            peak_amplitude=peak,
            rms_amplitude=rms,
            timestamp=self._get_timestamp(),
        )

    def write_wav(
        self,
        filepath: str,
        L: np.ndarray,
        R: np.ndarray,
        n_channels: int,
    ) -> None:
        """Write a WAV file."""
        with wave.open(filepath, 'w') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)

            if n_channels == 2:
                # Interleave stereo
                interleaved = np.empty(len(L) * 2, dtype=np.int16)
                interleaved[0::2] = L
                interleaved[1::2] = R
                wav.writeframes(interleaved.tobytes())
            else:
                # Mono
                wav.writeframes(L.tobytes())

    def write_audio(
        self,
        filename: str,
        L: Union[List[int], np.ndarray],
        R: Union[List[int], np.ndarray, None] = None,
        format: str = "wav",
        timestamp: bool = True,
        subdirectory: str = None,
        verbose: bool = True,
    ) -> AudioMetadata:
        """
        Write audio to file.

        filename: Base filename (without extension or timestamp)
        L: Left channel (or mono) samples
        R: Right channel samples (None for mono)
        format: Output format ("wav")
        timestamp: Whether to add timestamp to filename
        subdirectory: Optional subdirectory within output_dir
        verbose: Print output info

        Returns: AudioMetadata with file information
        """
        # Build filepath
        filepath = self._build_filepath(filename, format, timestamp, subdirectory)

        # Normalize audio data
        L_arr, R_arr, n_channels = self._normalize_audio(L, R)

        # Write based on format
        if format.lower() == "wav":
            self.write_wav(filepath, L_arr, R_arr, n_channels)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Compute metadata
        metadata = self._compute_metadata(filepath, L_arr, R_arr, n_channels, format)

        if verbose:
            self._print_info(metadata)

        return metadata

    def _print_info(self, meta: AudioMetadata) -> None:
        """Print info about written file."""
        peak_pct = 100 * meta.peak_amplitude / Q15_SCALE
        print(f"Wrote {meta.filepath}")
        print(f"  Duration: {meta.duration_sec:.2f}s | Channels: {meta.channels} | Rate: {meta.sample_rate}Hz")
        print(f"  Peak: {meta.peak_amplitude:.0f} ({peak_pct:.1f}%) | RMS: {meta.rms_amplitude:.0f}")


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# Global writer instance (lazy initialization)
_default_writer: Optional[AudioWriter] = None


def get_writer() -> AudioWriter:
    """Get the default AudioWriter instance."""
    global _default_writer
    if _default_writer is None:
        _default_writer = AudioWriter()
    return _default_writer


def set_output_dir(output_dir: str) -> None:
    """Set the default output directory."""
    global _default_writer
    _default_writer = AudioWriter(output_dir=output_dir)


def write_audio(
    filename: str,
    L: Union[List[int], np.ndarray],
    R: Union[List[int], np.ndarray, None] = None,
    format: str = "wav",
    timestamp: bool = True,
    subdirectory: str = None,
    verbose: bool = True,
) -> AudioMetadata:
    """
    Write audio to file using the default writer.

    This is the main entry point for audio output in the pipeline.

    Usage:
        from audio_io import write_audio

        # With timestamp (default)
        write_audio("my_sound", L, R)
        # Creates: audio_out/my_sound-20251219-114532.wav

        # Without timestamp
        write_audio("my_sound", L, R, timestamp=False)
        # Creates: audio_out/my_sound.wav

        # In subdirectory
        write_audio("test1", L, R, subdirectory="tests")
        # Creates: audio_out/tests/test1-20251219-114532.wav

        # Mono
        write_audio("mono_sound", mono_samples)
        # Creates mono WAV

    Returns: AudioMetadata with file information
    """
    return get_writer().write_audio(
        filename, L, R, format, timestamp, subdirectory, verbose
    )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO I/O MODULE TEST")
    print("=" * 60)
    print()

    # Generate test tone (440Hz sine wave, 1 second)
    duration = 1.0
    sample_rate = 48000
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    freq = 440.0
    amplitude = 16000

    # Stereo test tone with slight detune for width
    L = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    R = (amplitude * np.sin(2 * np.pi * freq * 1.002 * t)).astype(np.int16)

    # Write with timestamp
    print("Writing test tone with timestamp...")
    meta1 = write_audio("test_tone", L, R)
    print()

    # Write without timestamp
    print("Writing test tone without timestamp...")
    meta2 = write_audio("test_tone_no_ts", L, R, timestamp=False)
    print()

    # Write to subdirectory
    print("Writing to subdirectory...")
    meta3 = write_audio("subdir_test", L, R, subdirectory="test_outputs")
    print()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
