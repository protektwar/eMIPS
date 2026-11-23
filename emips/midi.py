"""MIDI notes as sound (PLAN M27 phase S4): the notes MARS's / RARS's
services 31 / 33 record (:class:`emips.console.MidiEvent`: the core's
cycle, pitch, duration, instrument, volume) — listed, rendered to a WAV
with Python's ``wave`` and a small additive synthesizer (one timbre and
envelope per General MIDI family: no SoundFont, no dependency), or
written as a Standard MIDI File (format 0, one tick = 1 ms) that any
synthesizer plays with its real instruments.

Time is the services' (``rvservices.CYCLES_PER_MS``: the cycle count at
1 MHz), so a run's tune is the same every time.  MARS plays a note on the
Java synthesizer when the service runs; 33 waits its duration, 31 does not
(both already so on the clock: phase S1)."""

from __future__ import annotations

import array
import io
import math
import struct
import wave

from .rvservices import CYCLES_PER_MS

RATE = 22050                                   # samples a second (mono, 16 bits)
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
#: the General MIDI families (instrument // 8): MARS's instrument numbers
FAMILIES = ("piano", "chromatic percussion", "organ", "guitar", "bass", "strings",
            "ensemble", "brass", "reed", "pipe", "synth lead", "synth pad",
            "synth effects", "ethnic", "percussive", "sound effects")
#: family -> (harmonic amplitudes, attack s, decay s, sustain level, release s,
#: percussive: the level falls all along the note)
TIMBRES = {
    0: ((1, .5, .3, .15, .1), .005, .3, .35, .15, True),          # piano
    1: ((1, 0, .4, 0, .2), .002, .5, .2, .3, True),               # chromatic perc.
    2: ((1, .7, .5, .4, .3, .2), .01, .05, .9, .05, False),       # organ
    3: ((1, .6, .4, .3, .2, .1), .003, .4, .3, .2, True),         # guitar
    4: ((1, .4, .1), .01, .2, .7, .1, False),                     # bass
    5: ((1, .5, .33, .25, .2, .16), .08, .1, .85, .2, False),     # strings
    6: ((1, .5, .33, .25, .2), .1, .1, .8, .3, False),            # ensemble
    7: ((1, .8, .6, .5, .4, .3), .03, .1, .8, .1, False),         # brass
    8: ((1, 0, .5, 0, .3, 0, .2), .03, .05, .85, .08, False),     # reed
    9: ((1, .1, .05), .05, .05, .9, .1, False),                   # pipe
    10: ((1, 0, .33, 0, .2, 0, .14), .005, .05, .8, .05, False),  # synth lead
    11: ((1, .3, .1), .3, .2, .8, .4, False),                     # synth pad
    12: ((1, .5, .5, .5), .05, .3, .5, .3, False),                # synth effects
    13: ((1, .3, .6, .2), .005, .3, .3, .2, True),                # ethnic
    14: ((1, .2), .001, .15, .1, .1, True),                       # percussive
    15: ((1, .7, .7, .7, .7), .01, .2, .5, .2, False),            # sound effects
}


def note_name(pitch: int) -> str:
    """60 = C4 (middle C), 69 = A4 (440 Hz)."""
    return f"{NAMES[pitch % 12]}{pitch // 12 - 1}"


def frequency(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def start_ms(ev) -> float:
    return ev.cycle / CYCLES_PER_MS


def lines(events) -> list[str]:
    """The debugger's ``midi``: one line a note."""
    if not events:
        return ["no MIDI notes yet (MARS / RARS services 31 and 33: $a0 pitch, $a1 "
                "duration in ms, $a2 instrument, $a3 volume)"]
    out = [f"{len(events)} note{'s' if len(events) != 1 else ''}:"]
    for ev in events:
        fam = FAMILIES[ev.instrument // 8]
        out.append(f"  {start_ms(ev):10.3f} ms  core {ev.cpu}  {note_name(ev.pitch):<4} "
                   f"(pitch {ev.pitch:3})  {ev.duration} ms  instrument {ev.instrument} "
                   f"({fam})  volume {ev.volume}" + ("  [33: waited]" if ev.sync else ""))
    return out


def _table(harmonics, size: int = 2048) -> list[float]:
    total = sum(harmonics)
    return [sum(a * math.sin(2 * math.pi * (k + 1) * i / size)
                for k, a in enumerate(harmonics)) / total for i in range(size)]


_TABLES: dict[int, list[float]] = {}


def _note(ev, rate: int) -> tuple[int, array.array]:
    """One note's samples (floats) and where they start (in samples)."""
    fam = ev.instrument // 8
    harmonics, attack, decay, sustain, release, percussive = TIMBRES[fam]
    table = _TABLES.get(fam)
    if table is None:
        table = _TABLES[fam] = _table(harmonics)
    size = len(table)
    held = ev.duration / 1000
    n = int((held + release) * rate)
    step = frequency(ev.pitch) * size / rate
    amp = ev.volume / 127 * 0.5
    out = array.array("d", bytes(8 * n))
    phase = 0.0
    for i in range(n):
        t = i / rate
        if t < attack:
            env = t / attack
        elif t < attack + decay:
            env = 1 - (1 - sustain) * (t - attack) / decay
        else:
            env = sustain
        if percussive:
            env *= math.exp(-3 * max(t - attack, 0))
        if t > held:
            env *= max(0.0, 1 - (t - held) / release)
        out[i] = amp * env * table[int(phase) % size]
        phase += step
    return int(start_ms(ev) * rate / 1000), out


def render(events, rate: int = RATE) -> array.array:
    """Every note mixed: 16-bit samples (scaled down together when the mix
    would clip)."""
    notes = [_note(ev, rate) for ev in events if ev.duration > 0]
    length = max((s + len(x) for s, x in notes), default=0)
    mix = array.array("d", bytes(8 * length))
    for s, x in notes:
        for i, v in enumerate(x):
            mix[s + i] += v
    peak = max((abs(v) for v in mix), default=0.0)
    k = 32767 * (0.95 / peak if peak > 0.95 else 1.0)
    return array.array("h", (int(round(v * k)) for v in mix))


def wav(events, rate: int = RATE) -> bytes:
    """A WAV file (mono, 16 bits, ``rate`` Hz) of the notes."""
    samples = render(events, rate)
    if samples.itemsize == 2 and struct.pack("=h", 1) != struct.pack("<h", 1):
        samples.byteswap()                     # (WAV is little-endian)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def midi_file(events) -> bytes:
    """A Standard MIDI File (format 0): 1000 ticks a quarter note at 60 000
    000 / 1 000 000 = 60 BPM, so one tick = 1 ms; each instrument its own
    channel (the first 15 instruments; channel 9 = drums is skipped), with
    its program change first."""
    channels: dict[int, int] = {}
    free = [c for c in range(16) if c != 9]
    timeline: list[tuple[int, int, bytes]] = []          # (ms, order, message)
    for ev in events:
        ch = channels.get(ev.instrument)
        if ch is None:
            ch = channels[ev.instrument] = free.pop(0) if free else 15
            timeline.append((0, 0, bytes((0xC0 | ch, ev.instrument))))
        on = int(round(start_ms(ev)))
        timeline.append((on, 2, bytes((0x90 | ch, ev.pitch, ev.volume))))
        timeline.append((on + ev.duration, 1, bytes((0x80 | ch, ev.pitch, 0))))
    timeline.sort(key=lambda e: (e[0], e[1]))
    track = bytearray(b"\x00\xff\x51\x03" + (1_000_000).to_bytes(3, "big"))
    now = 0
    for ms, _, msg in timeline:
        track += _vlq(ms - now) + msg
        now = ms
    track += b"\x00\xff\x2f\x00"
    return (b"MThd" + struct.pack(">IHHH", 6, 0, 1, 1000)
            + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
