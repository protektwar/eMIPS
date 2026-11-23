"""The Sound pane (PLAN M27 phase S4): the MIDI notes the program played
(MARS's / RARS's services 31 / 33, :meth:`Session.midi_events`) — when on
the cycle clock, the core, the note, its duration, instrument (and General
MIDI family) and volume — with Save WAV (:func:`emips.midi.wav`, a simple
synthesizer) and Save MIDI (:func:`emips.midi.midi_file`).  When the Qt
binding has QtMultimedia, *Play while running* sounds each new note as the
program plays it (rendered by :mod:`emips.midi`); without it, the list."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .. import midi as MIDI
from ..session import Session
from .i18n import N_, tr
from .qt import QtCore, QtWidgets, Signal, multimedia

COLUMNS = (N_("time (ms)"), N_("core"), N_("note"), N_("duration (ms)"), N_("instrument"),
           N_("volume"), N_("33"))


class SoundView(QtWidgets.QWidget):
    #: Save WAV / Save MIDI: "wav" | "mid"
    save_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.note = QtWidgets.QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.mm = multimedia()
        self.play = QtWidgets.QCheckBox(tr("Play while running"))
        self.play.setEnabled(self.mm is not None)
        self.play.setChecked(self.mm is not None)
        self.play.setToolTip(tr("Sound each note as the program plays it") if self.mm else
                             tr("Needs QtMultimedia (not installed with this Qt binding): "
                                "Save WAV or Save MIDI and play the file"))
        row.addWidget(self.play)
        row.addStretch()
        self.save_wav = QtWidgets.QPushButton(tr("Save WAV…"))
        self.save_wav.clicked.connect(lambda: self.save_requested.emit("wav"))
        row.addWidget(self.save_wav)
        self.save_mid = QtWidgets.QPushButton(tr("Save MIDI…"))
        self.save_mid.clicked.connect(lambda: self.save_requested.emit("mid"))
        row.addWidget(self.save_mid)
        lay.addLayout(row)
        self.played = 0                     # the notes already sounded
        self._effects: list = []
        self._dir: tempfile.TemporaryDirectory | None = None

    def refresh(self, s: Session) -> None:
        events = s.midi_events() if s.machine is not None else []
        self.note.setText(tr("{n} MIDI notes (services 31 / 33: $a0 pitch, $a1 ms, $a2 "
                             "instrument, $a3 volume; times on the cycle clock, 1 ms = "
                             "1000 cycles)").format(n=len(events)) if events else
                          tr("No MIDI notes yet: services 31 (play) and 33 (play and wait) "
                             "record them here."))
        if len(events) < self.table.rowCount():          # a reset / step back
            self.played = min(self.played, len(events))
        self.table.setRowCount(len(events))
        for r, ev in enumerate(events):
            cells = (f"{MIDI.start_ms(ev):.3f}", str(ev.cpu),
                     f"{MIDI.note_name(ev.pitch)} ({ev.pitch})", str(ev.duration),
                     f"{ev.instrument} ({tr(MIDI.FAMILIES[ev.instrument // 8])})",
                     str(ev.volume), "✓" if ev.sync else "")
            for c, text in enumerate(cells):
                item = self.table.item(r, c)
                if item is None or item.text() != text:
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        for w in (self.save_wav, self.save_mid):
            w.setEnabled(bool(events))
        if self.play.isChecked() and self.mm is not None:
            for ev in events[self.played:]:
                self._sound(ev)
        self.played = len(events)

    def _sound(self, ev) -> None:
        from dataclasses import replace
        if self._dir is None:
            self._dir = tempfile.TemporaryDirectory(prefix="emips-sound-")
        path = Path(self._dir.name) / f"note{len(self._effects) % 64}.wav"
        path.write_bytes(MIDI.wav([replace(ev, cycle=0)]))
        effect = self.mm.QSoundEffect(self)
        effect.setSource(QtCore.QUrl.fromLocalFile(str(path)))
        effect.play()
        self._effects = [e for e in self._effects if e.isPlaying()][-32:] + [effect]
