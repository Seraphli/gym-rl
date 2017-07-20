import lzma, pickle, multiprocessing, zlib


def compress(data):
    return zlib.compress(pickle.dumps(data))


def decompress(data):
    return pickle.loads(zlib.decompress(data))


def split_block(datum, length):
    return [datum[i:i + length] for i in range(0, len(datum), length)]


def compress_mp(datum, threads=multiprocessing.cpu_count()):
    datum = pickle.dumps(datum)
    block_size = int(round(len(datum) / (threads - 1)))
    data = split_block(datum, block_size)
    pool = multiprocessing.Pool(threads)
    compressed = pool.map(lzma.compress, data)
    return lzma.compress(pickle.dumps(compressed))


def decompress_mp(data, threads=multiprocessing.cpu_count()):
    data = pickle.loads(lzma.decompress(data))
    pool = multiprocessing.Pool(threads)
    decompressed = pool.map(lzma.decompress, data)
    return pickle.loads(b"".join(decompressed))
