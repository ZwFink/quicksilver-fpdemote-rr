"""Pure-Python reader for Mneme snapshot files (bytes layout v0, diff layout v1).

The layouts mirror include/mneme/MnemeSnapshotFormat.hpp at Mneme d0d799d. A
snapshot is read into a MemoryImage: the device allocations (blobs) keyed by
their recorded device address, the device globals, and the kernel arguments.
"""

import bisect
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

MAGIC = b"MNEMESNP"
LEGACY_DIFF_MAGIC = b"MNEME_DIFF_V1"
KIND_BYTES, KIND_DIFF = 1, 2


@dataclass
class Blob:
    addr: int
    actual_size: int
    data: np.ndarray  # uint8, len == recorded size
    metadata: Tuple  # (builtin, threshold, threshold_kind, norm, tag)


@dataclass
class Global:
    name: str
    addr: int
    data: np.ndarray


@dataclass
class MemoryImage:
    blobs: Dict[int, Blob] = field(default_factory=dict)
    globals: Dict[str, Global] = field(default_factory=dict)
    args: List[bytes] = field(default_factory=list)
    _starts: Optional[List[int]] = None

    def copy(self) -> "MemoryImage":
        return MemoryImage(
            blobs={a: Blob(b.addr, b.actual_size, b.data.copy(), b.metadata) for a, b in self.blobs.items()},
            globals={n: Global(g.name, g.addr, g.data.copy()) for n, g in self.globals.items()},
            args=list(self.args),
        )

    def _index(self):
        if self._starts is None or len(self._starts) != len(self.blobs):
            self._starts = sorted(self.blobs)
        return self._starts

    def find(self, addr: int, size: int = 1) -> Tuple[Blob, int]:
        """Return the blob holding [addr, addr+size) and the offset of addr in it."""
        starts = self._index()
        i = bisect.bisect_right(starts, addr) - 1
        if i < 0:
            raise KeyError(f"address {addr:#x} is below every recorded blob")
        blob = self.blobs[starts[i]]
        off = addr - blob.addr
        if off + size > len(blob.data):
            raise KeyError(
                f"range [{addr:#x}, +{size}) is not inside a recorded blob "
                f"(nearest blob {blob.addr:#x} size {len(blob.data)})"
            )
        return blob, off

    def read(self, addr: int, size: int) -> np.ndarray:
        blob, off = self.find(addr, size)
        return blob.data[off : off + size]

    def u64(self, addr: int) -> int:
        return int(self.read(addr, 8).view(np.uint64)[0])

    def i32(self, addr: int) -> int:
        return int(self.read(addr, 4).view(np.int32)[0])

    def array(self, addr: int, dtype, count: int) -> np.ndarray:
        dtype = np.dtype(dtype)
        if count == 0:
            return np.zeros(0, dtype=dtype)
        return self.read(addr, dtype.itemsize * count).view(dtype)


class _Cursor:
    def __init__(self, buf: memoryview, pos: int):
        self.buf, self.pos = buf, pos

    def scalar(self, fmt: str):
        (v,) = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return v

    def u64(self) -> int:
        return self.scalar("Q")

    def take(self, n: int) -> np.ndarray:
        out = np.frombuffer(self.buf, dtype=np.uint8, count=n, offset=self.pos).copy()
        self.pos += n
        return out

    def string(self) -> str:
        n = self.u64()
        s = bytes(self.buf[self.pos : self.pos + n]).decode()
        self.pos += n
        return s

    def metadata(self) -> Tuple:
        builtin = self.scalar("B")
        threshold = self.scalar("d")
        kind = self.scalar("B")
        norm = self.scalar("B")
        tag_len = self.u64()
        tag = bytes(self.buf[self.pos : self.pos + tag_len]).decode() if tag_len else None
        self.pos += tag_len
        return (builtin, threshold, kind, norm, tag)

    def global_header(self) -> Tuple[str, int, int]:
        name = self.string()
        size = self.u64()
        addr = self.u64()
        return name, size, addr

    def blob_header(self) -> Tuple[int, int, int]:
        actual = self.u64()
        size = self.u64()
        addr = self.u64()
        return actual, size, addr

    def apply_ranges(self, target: np.ndarray) -> List[Tuple[int, int]]:
        ranges = []
        for _ in range(self.u64()):
            off = self.u64()
            size = self.u64()
            if off + size > len(target):
                raise ValueError(f"diff range {off}+{size} exceeds target size {len(target)}")
            target[off : off + size] = self.take(size)
            ranges.append((off, size))
        return ranges


def _header(buf: memoryview) -> Tuple[int, int, int]:
    if len(buf) >= 16 and bytes(buf[:8]) == MAGIC:
        kind, version = struct.unpack_from("<II", buf, 8)
        return kind, version, 16
    if bytes(buf[: len(LEGACY_DIFF_MAGIC)]) == LEGACY_DIFF_MAGIC:
        return KIND_DIFF, 1, len(LEGACY_DIFF_MAGIC)
    return KIND_BYTES, 0, 0


def read_snapshot(path: str, base: Optional[MemoryImage] = None) -> MemoryImage:
    """Read a snapshot. A diff snapshot needs the image of its base prologue."""
    with open(path, "rb") as f:
        buf = memoryview(f.read())
    kind, version, pos = _header(buf)
    c = _Cursor(buf, pos)

    if (kind, version) == (KIND_BYTES, 0):
        img = MemoryImage()
        for _ in range(c.u64()):
            name, size, addr = c.global_header()
            img.globals[name] = Global(name, addr, c.take(size))
        for _ in range(c.u64()):
            actual, size, addr = c.blob_header()
            data = c.take(size)
            img.blobs[addr] = Blob(addr, actual, data, c.metadata())
        for _ in range(c.u64()):
            img.args.append(bytes(c.take(c.u64())))
    elif (kind, version) == (KIND_DIFF, 1):
        if base is None:
            raise ValueError(f"{path} is a diff snapshot and needs its base prologue")
        img = base.copy()
        n = c.u64()
        if n != len(img.globals):
            raise ValueError(f"{path}: diff has {n} globals, base has {len(img.globals)}")
        for _ in range(n):
            name, size, addr = c.global_header()
            g = img.globals[name]
            if len(g.data) != size:
                raise ValueError(f"{path}: global {name} size mismatch")
            g.addr = addr
            c.apply_ranges(g.data)
        n = c.u64()
        if n != len(img.blobs):
            raise ValueError(f"{path}: diff has {n} blobs, base has {len(img.blobs)}")
        for _ in range(n):
            actual, size, addr = c.blob_header()
            blob = img.blobs[addr]
            if blob.actual_size != actual or len(blob.data) != size:
                raise ValueError(f"{path}: blob {addr:#x} size mismatch")
            blob.metadata = c.metadata()
            c.apply_ranges(blob.data)
    else:
        raise ValueError(f"{path}: unsupported snapshot kind {kind} version {version}")

    if c.pos != len(buf):
        raise ValueError(f"{path}: {len(buf) - c.pos} trailing bytes after parsing")
    return img


def save_image(path: str, img: MemoryImage) -> None:
    """Save an image as .npz (blob_<addr hex>, global_<name>, arg_<i>)."""
    arrays = {f"blob_{a:x}": b.data for a, b in img.blobs.items()}
    arrays.update({f"global_{n}": g.data for n, g in img.globals.items()})
    arrays.update({f"gaddr_{n}": np.array([g.addr], dtype=np.uint64) for n, g in img.globals.items()})
    arrays.update({f"arg_{i}": np.frombuffer(a, dtype=np.uint8) for i, a in enumerate(img.args)})
    np.savez(path, **arrays)


def load_image(path: str) -> MemoryImage:
    img = MemoryImage()
    with np.load(path) as z:
        args = {}
        for k in z.files:
            if k.startswith("blob_"):
                a = int(k[5:], 16)
                img.blobs[a] = Blob(a, 0, z[k].copy(), ())
            elif k.startswith("global_"):
                n = k[len("global_"):]
                img.globals[n] = Global(n, int(z["gaddr_" + n][0]), z[k].copy())
            elif k.startswith("arg_"):
                args[int(k[4:])] = bytes(z[k])
        img.args = [args[i] for i in sorted(args)]
    return img
