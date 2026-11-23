"""Files for MARS / RARS programs without the host's file system (PLAN M27
phase S3): MARS 13-16 (open, read, write, close) and RARS's 1024 / 63 /
64 / 57 / 62 (open, read, write, close, lseek) work on a **file area**
given with the program, the way the Malta's IDE disks use their images
(``emips.ata.Media``): ``--file NAME=PATH`` (repeatable; NAME defaults to
the PATH's file name) puts PATH's bytes in the area under NAME; a program
opens NAME, reads it, writes new files or appends to given ones.  Every
reset starts from the given files again, so a run replays exactly (step
back, checks); what a run wrote stays in the area until then, for the
debugger's ``files`` / the window's Files pane and their Save.  A program
never reaches any other file.

The descriptor table is MARS's / RARS's ``SystemIO.FileIOData``
(MARS 4.5's ``mars/util/SystemIO.java``, RARS 1.6's ``rars/util/SystemIO.
java``): 32 descriptors, 0-2 the standard streams (the services' input and
output), the lowest free one from 3; flags 0 = read, 1 = write (the file
is emptied), 9 = write appending, anything else -1; a name already open
-1; read returns the bytes read, 0 at the end, -1 on a descriptor not open
for reading; write -1 on one not open for writing; close of 0-2 or a free
descriptor does nothing.  Their quirk is kept: opening a file that does
not exist leaves the name in the table — its descriptor stays taken and
the name "already open" until it is closed (``nowOpening`` fills the slot
before the open fails).  RARS's lseek (62: a0 fd, a1 offset, a2 whence 0 /
1 / 2) works only on a descriptor open for reading, as RARS's ``seek``.
"""

from __future__ import annotations

from pathlib import Path

MAX_FILES = 32                              # SYSCALL_MAXFILES
O_RDONLY, O_WRONLY, O_APPEND = 0, 1, 8
STDIO = (0, 1, 2)


def parse_files(text: str) -> list[tuple[str, str]]:
    """``NAME=PATH,...`` (or ``PATH``: NAME = its file name) -> pairs;
    ValueError for an empty name or path or a name given twice."""
    out: list[tuple[str, str]] = []
    for part in (p.strip() for p in (text or "").split(",")):
        if not part:
            continue
        name, eq, path = part.partition("=")
        if not eq:
            name, path = Path(part).name, part
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"--file {part!r}: NAME=PATH (or PATH)")
        if any(n == name for n, _ in out):
            raise ValueError(f"--file: {name!r} given twice")
        out.append((name, path))
    return out


class FileArea:
    """The program's files: the given ones (name -> bytes, never changed)
    and what the run did to them (:attr:`data`: name -> bytearray; a name
    a run wrote is in :attr:`written`)."""

    def __init__(self, given: dict[str, bytes] | None = None) -> None:
        self.given = dict(given or {})
        self.data: dict[str, bytearray] = {k: bytearray(v) for k, v in self.given.items()}
        self.written: set[str] = set()

    def state(self, name: str) -> str:
        if name not in self.given:
            return "created"
        return "written" if name in self.written else "given"

    def lines(self) -> list[str]:
        if not self.data:
            return ["no files: give some with --file NAME=PATH (or set files / Settings "
                    "> Exceptions & I/O > Files)"]
        return [f"{name}: {len(self.data[name])} bytes ({self.state(name)})"
                for name in sorted(self.data)]


class FileTable:
    """MARS's / RARS's descriptor table over a :class:`FileArea` (the
    standard streams 0-2 are the services' own: :meth:`stdio`)."""

    def __init__(self, area: FileArea) -> None:
        self.area = area
        self.names: list[str | None] = [None] * MAX_FILES
        self.flags = [-1] * MAX_FILES
        self.ok = [False] * MAX_FILES           # (the "stream" was opened)
        self.pos = [0] * MAX_FILES
        for fd, name, flag in ((0, "STDIN", O_RDONLY), (1, "STDOUT", O_WRONLY),
                               (2, "STDERR", O_WRONLY)):
            self.names[fd], self.flags[fd], self.ok[fd] = name, flag, True
        self.error = "File operation OK"

    def open(self, name: str, flags: int) -> int:
        if name in self.names:
            self.error = f"File name {name} is already open."
            return -1
        if flags not in (O_RDONLY, O_WRONLY, O_WRONLY | O_APPEND):
            self.error = f"File name {name} has unknown requested opening flag"
            return -1
        fd = next((i for i in range(MAX_FILES) if self.names[i] is None), -1)
        if fd < 0:
            self.error = f"File name {name} exceeds maximum open file limit of {MAX_FILES}"
            return -1
        self.names[fd], self.flags[fd] = name, flags   # (kept even when the open fails)
        self.error = "File operation OK"
        area = self.area
        if flags == O_RDONLY:
            if name not in area.data:
                self.error = f"File {name} not found, open for input."
                return -1
            self.pos[fd] = 0
        else:
            if flags & O_APPEND and name in area.data:
                self.pos[fd] = len(area.data[name])
            else:
                area.data[name] = bytearray()
                area.written.add(name)
                self.pos[fd] = 0
        self.ok[fd] = True
        return fd

    def _usable(self, fd: int, flag: int) -> bool:
        if not 0 <= fd < MAX_FILES or self.names[fd] is None or not self.ok[fd]:
            return False
        if flag == O_RDONLY:
            return self.flags[fd] == O_RDONLY
        return bool(self.flags[fd] & O_WRONLY)

    def read(self, fd: int, length: int) -> bytes | None:
        """Up to ``length`` bytes (b"" at the end); None: not open for reading."""
        if not self._usable(fd, O_RDONLY):
            self.error = f"File descriptor {fd} is not open for reading"
            return None
        data = self.area.data[self.names[fd]]
        out = bytes(data[self.pos[fd]:self.pos[fd] + max(length, 0)])
        self.pos[fd] += len(out)
        return out

    def write(self, fd: int, data: bytes) -> int:
        if not self._usable(fd, O_WRONLY):
            self.error = f"File descriptor {fd} is not open for writing"
            return -1
        name = self.names[fd]
        buf = self.area.data.setdefault(name, bytearray())
        p = self.pos[fd]
        buf[p:p + len(data)] = data
        self.pos[fd] = p + len(data)
        self.area.written.add(name)
        return len(data)

    def close(self, fd: int) -> None:
        if fd <= 2 or fd >= MAX_FILES:
            return
        self.names[fd], self.flags[fd], self.ok[fd], self.pos[fd] = None, -1, False, 0

    def seek(self, fd: int, offset: int, whence: int) -> int:
        """RARS's lseek: only a descriptor open for reading (as RARS)."""
        if not self._usable(fd, O_RDONLY) or fd in STDIO:
            return -1
        base = {0: 0, 1: self.pos[fd], 2: len(self.area.data[self.names[fd]])}.get(whence)
        if base is None or base + offset < 0:
            return -1
        self.pos[fd] = base + offset
        return self.pos[fd]

    def open_lines(self) -> list[str]:
        return [f"fd {fd}: {self.names[fd]} ({'read' if self.flags[fd] == O_RDONLY else 'append' if self.flags[fd] & O_APPEND else 'write'}"
                + (f", at {self.pos[fd]}" if self.ok[fd] else ", failed to open") + ")"
                for fd in range(3, MAX_FILES) if self.names[fd] is not None]


def dump_bytes(data: bytes, limit: int = 4096) -> list[str]:
    """A file's contents for the views: its text when it is printable text
    (tabs and newlines allowed), else a hex dump; at most ``limit`` bytes."""
    shown = data[:limit]
    more = [f"... ({len(data) - limit} more bytes)"] if len(data) > limit else []
    try:
        text = shown.decode("ascii")
    except UnicodeDecodeError:
        text = None
    if text is not None and all(c.isprintable() or c in "\t\n\r" for c in text):
        return (text.splitlines() or [""]) + more if text else ["(empty)"]
    out = []
    for off in range(0, len(shown), 16):
        row = shown[off:off + 16]
        out.append(f"{off:06x}  {row.hex(' '):<47}  "
                   + "".join(chr(b) if 32 <= b < 127 else "." for b in row))
    return out + more
