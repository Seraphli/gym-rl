import numpy as np, multiprocessing as mp
from util.compress import *


class MemoryBlock(object):
    def __init__(self, shape, block_size):
        self._shape = shape
        self._block_size = block_size
        self._actual_size = 0
        self.archived = False
        self._buffer = []
        self._allocate()

    def _allocate(self):
        for i in range(self._shape[0]):
            shape, dtype = self._shape[i + 1]
            self._buffer.append(np.empty((self._block_size,) + shape, dtype=dtype))

    def _finalize(self):
        for i in range(self._shape[0]):
            self._buffer[i] = self._buffer[i][:self._actual_size, ...]

    def append(self, data):
        if self.is_full():
            return False
        for i in range(self._shape[0]):
            self._buffer[i][self._actual_size, ...] = data[i]
        self._actual_size += 1
        return True

    def archive(self):
        self._finalize()
        self.archived = True
        return compress(self)

    @staticmethod
    def restore(data):
        obj = decompress(data)
        obj.archived = False
        return obj

    def is_full(self):
        return self._actual_size == self._block_size

    def __len__(self):
        return self._actual_size

    def __getitem__(self, i):
        assert i < self._actual_size, 'Out of range'
        return [self._buffer[idx][i] for idx in range(self._shape[0])]


class Memory(object):
    def __init__(self, size, shape, block_size=1000):
        self._size = size
        self._shape = shape
        self._block_size = block_size
        self._memory = []
        self._current_block = None
        if self._size != 0:
            assert self._size % self._block_size == 0
            self._blocks_n = self._size // self._block_size
        self._blocks_count = 0
        self._actual_size = 0
        self._mpp_size = 6
        self.mpp = mp.Pool(self._mpp_size)
        self._memory_process_lock = mp.Lock()
        self._allocate()

    def __getstate__(self):
        s = self.__dict__.copy()
        s.pop('mpp')
        return s

    def __setstate__(self, state):
        self.__dict__ = state
        self.mpp = mp.Pool(self._mpp_size)

    @staticmethod
    def _archive_current(memory, current_block):
        if current_block:
            memory.append(current_block.archive())
        return memory

    def _mp_callback(self, r):
        self._memory = r
        self._memory_process_lock.release()

    def _mp_archive_current(self):
        self._memory_process_lock.acquire()
        self.mpp.apply_async(Memory._archive_current, (self._memory, self._current_block), callback=self._mp_callback)

    def _allocate(self):
        self._mp_archive_current()
        self._current_block = MemoryBlock(self._shape, self._block_size)
        self._blocks_count += 1
        # We will have one normal block and self._blocks_n archived memory blocks.
        if self._size != 0 and self._blocks_count == self._blocks_n + 2:
            del self._memory[0]
            self._blocks_count -= 1
            self._actual_size -= self._block_size

    def archive_memory(self):
        self.mpp.close()
        self.mpp.join()
        self._memory_process_lock.acquire()
        self._memory.append(self._current_block.archive())
        self._memory_process_lock.release()
        self._current_block = None

    def restore_memory(self):
        self._current_block = MemoryBlock.restore(self._memory.pop(-1))

    def archive_block(self, i=None):
        if not i:
            for _ in range(len(self._memory)):
                if self._memory[_].archived == False:
                    self._memory[_] = self._memory[_].archive()
        else:
            if self._memory[i].archived == False:
                self._memory[i] = self._memory[i].archive()

    def restore_block(self, i=None):
        if not i:
            for _ in range(len(self._memory)):
                self._memory[_] = MemoryBlock.restore(self._memory[_])
        else:
            self._memory[i] = MemoryBlock.restore(self._memory[i])

    def save(self, filename):
        self.archive_memory()
        with open(filename, 'wb') as f:
            f.write(compress(self))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            obj = decompress(f.read())
            obj.restore_memory()
            return obj

    def is_full(self):
        if self._size == 0:
            return False
        return self._blocks_count == self._blocks_n

    def __len__(self):
        return self._actual_size

    def block_len(self):
        return len(self._memory)

    def __getitem__(self, i):
        assert i < self._actual_size, 'Out of range'
        block_i = i // self._block_size
        i -= block_i * self._block_size
        if block_i + 1 < self._blocks_count:
            if self._memory[block_i].archived:
                return MemoryBlock.restore(self._memory[block_i])[i]
            else:
                return self._memory[block_i][i]
        else:
            return self._current_block[i]

    def append(self, data):
        if self._current_block.is_full():
            self._allocate()
        self._current_block.append(data)
        self._actual_size += 1

    def sample_enqueue(self, batch_size):
        self._batch_size = batch_size
        manager = multiprocessing.Manager()
        self._sample_queue = manager.Queue(100)
        for _ in range(2):
            block_index = np.random.randint(0, len(self._memory))
            block = MemoryBlock.restore(self._memory[block_index])
            self.mpp.apply_async(Memory.mp_enqueue,
                                 (self._sample_queue, block, self._block_size, self._batch_size),
                                 callback=self.restore_callback)

    @staticmethod
    def mp_enqueue(q, block, block_size, batch_size):
        samples = []
        for _ in range(10):
            while len(samples) < batch_size:
                sample_index = np.random.randint(block_size - 1)
                sample = block[sample_index]
                if sample[3]:
                    continue
                samples.append(sample + [block[sample_index + 1][0]])
            s, a, r, t, s_ = [[i[j] for i in samples] for j in range(5)]
            q.put((s, a, r, t, s_))

    def restore_callback(self, r):
        block_index = np.random.randint(0, len(self._memory))
        block = MemoryBlock.restore(self._memory[block_index])
        self.mpp.apply_async(Memory.mp_enqueue,
                             (self._sample_queue, block, self._block_size, self._batch_size),
                             callback=self.restore_callback)

    def sample(self):
        return self._sample_queue.get()
