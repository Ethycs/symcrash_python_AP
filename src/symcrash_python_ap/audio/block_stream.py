"""Adapt queued float PCM blocks to arbitrary device callback lengths."""
import numpy as np


class AudioBlockReader:
    def __init__(self, source, channels=1):
        self.source = source
        self.channels = channels
        self.pending = np.empty((0, channels), dtype=np.float32)
        self.offset = 0
        self.underruns = 0

    def fill(self, destination):
        """Preserve partial blocks; silence only a confirmed source underrun."""
        output = destination.reshape(-1, 1) if destination.ndim == 1 else destination
        if output.ndim != 2 or output.shape[1] != self.channels:
            raise ValueError('Device output channel count does not match the block reader')
        position = 0
        while position < len(output):
            if self.offset == len(self.pending):
                block = self.source()
                if block is None:
                    output[position:] = 0
                    self.underruns += 1
                    return
                block = np.asarray(block, dtype=np.float32)
                if block.ndim == 1:
                    block = block.reshape(-1, 1)
                if block.ndim != 2 or block.shape[1] != self.channels or len(block) == 0:
                    raise ValueError('Queued audio block has an invalid shape')
                self.pending, self.offset = block, 0
            count = min(len(output) - position, len(self.pending) - self.offset)
            output[position:position+count] = self.pending[self.offset:self.offset+count]
            position += count
            self.offset += count
