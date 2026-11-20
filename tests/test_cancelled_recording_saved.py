"""A converse exchange cancelled mid-recording keeps the captured speech.

Before this, cancelling (ESC, tool-call cancel) set the recording thread's
stop_event and re-raised; the thread's buffer was returned to a future that
nothing awaited, so up to listen_duration_max seconds of speech were lost
with no file and no log line. Now the recording thread writes the buffer to
AUDIO_DIR/YYYY/MM/<ts>_<conversation>_cancelled.wav and logs an
"stt_cancelled" exchange, while the cancel itself is re-raised at once.
"""

import asyncio
import json
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.io import wavfile

sys.modules.setdefault('webrtcvad', MagicMock())

from voice_mode.config import SAMPLE_RATE  # noqa: E402


def _start_listen(listen_and_transcribe, control_state):
    return asyncio.ensure_future(listen_and_transcribe(
        control_state=control_state,
        listen_duration_max=30.0,
        listen_duration_min=0.5,
        disable_silence_detection=False,
        vad_aggressiveness=None,
        chime_enabled=False,
        chime_leading_silence=None,
        chime_trailing_silence=None,
        transport="local",
        event_logger=None,
    ))


async def _cancel_mid_recording(record_fake):
    """Run listen_and_transcribe with record_fake, cancel it once recording
    has started, and return after the cancel has propagated."""
    from voice_mode.control_channel import get_control_state
    from voice_mode.tools.converse import listen_and_transcribe

    async def _noop_feedback(*_a, **_k):
        return None

    get_control_state().reset()
    try:
        with patch('voice_mode.tools.converse.play_audio_feedback', new=_noop_feedback), \
             patch('voice_mode.tools.converse.record_audio_with_silence_detection', new=record_fake):
            task = _start_listen(listen_and_transcribe, get_control_state())
            for _ in range(100):
                if record_fake.entered.is_set():
                    break
                await asyncio.sleep(0.02)
            assert record_fake.entered.is_set(), "recorder never started"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert record_fake.returned.wait(timeout=5.0), "recording thread never returned"
    finally:
        get_control_state().reset()


class _Recorder:
    """Stands in for record_audio_with_silence_detection: blocks like the VAD
    loop until the caller's cancel sets stop_event, then returns the
    "speech" captured so far (1.5 s of a 440 Hz tone)."""

    def __init__(self, seconds=1.5):
        self.entered = threading.Event()
        self.returned = threading.Event()
        t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
        self.audio = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)

    def __call__(self, max_duration, disable_silence_detection, min_duration,
                 vad_aggressiveness, stop_event=None):
        self.entered.set()
        assert stop_event is not None
        stop_event.wait(timeout=5.0)
        try:
            return self.audio, True
        finally:
            self.returned.set()


def _saved_wavs():
    import voice_mode.config as config
    return sorted(Path(config.AUDIO_DIR).rglob("*_cancelled.wav"))


def _exchange_lines():
    from voice_mode.conversation_logger import get_conversation_logger
    base = get_conversation_logger().base_dir
    lines = []
    for path in sorted(base.glob("exchanges_*.jsonl")):
        lines += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return lines


def _wait_for(check, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(0.02)
    return check()


@pytest.mark.asyncio
async def test_cancel_mid_recording_saves_wav_and_logs_it():
    recorder = _Recorder(seconds=1.5)
    await _cancel_mid_recording(recorder)

    wavs = _wait_for(_saved_wavs)
    assert len(wavs) == 1, f"expected one cancelled wav, found {wavs}"
    wav = wavs[0]

    # YYYY/MM/<ts>_<conversation>_cancelled.wav, and it plays back the buffer.
    assert wav.parent.parent.name.isdigit() and len(wav.parent.parent.name) == 4
    assert wav.parent.name.isdigit() and len(wav.parent.name) == 2
    rate, data = wavfile.read(wav)
    assert rate == SAMPLE_RATE
    assert np.array_equal(data, recorder.audio)

    logged = _wait_for(lambda: [e for e in _exchange_lines() if e.get("type") == "stt_cancelled"])
    assert len(logged) == 1
    assert logged[0]["audio_file"] == str(wav)
    assert logged[0]["duration_ms"] == 1500


@pytest.mark.asyncio
async def test_cancel_with_nothing_recorded_writes_nothing():
    recorder = _Recorder(seconds=0)
    await _cancel_mid_recording(recorder)

    assert _saved_wavs() == []
    assert [e for e in _exchange_lines() if e.get("type") == "stt_cancelled"] == []


@pytest.mark.asyncio
async def test_cancel_saving_can_be_turned_off():
    recorder = _Recorder(seconds=1.0)
    with patch('voice_mode.config.SAVE_CANCELLED_AUDIO', False):
        await _cancel_mid_recording(recorder)
        import time
        time.sleep(0.3)  # give the recording thread time to finish

    assert _saved_wavs() == []


@pytest.mark.asyncio
async def test_cancelled_recording_transcribed_to_heard_log_when_enabled():
    recorder = _Recorder(seconds=1.0)
    written = []

    async def _stt(audio_data, save_audio, audio_dir, transport):
        return {"text": "what I said before the cancel"}

    def _partial(text, **kw):
        written.append({"text": text, **kw})

    # The save runs on the recording thread after the cancel returns, so the
    # flag has to stay patched until the partial is written.
    with patch('voice_mode.tools.converse.speech_to_text', new=_stt), \
         patch('voice_mode.heard.partial', new=_partial), \
         patch('voice_mode.config.TRANSCRIBE_CANCELLED', True):
        await _cancel_mid_recording(recorder)
        assert _wait_for(lambda: written), "no heard partial was written"

    assert written[0]["text"] == "what I said before the cancel"
    assert written[0]["final"] is False
    assert written[0]["cancelled"] is True
    assert written[0]["audio_file"].endswith("_cancelled.wav")


@pytest.mark.asyncio
async def test_cancel_is_not_delayed_by_the_save():
    """The cancel re-raises at once; the save runs on the recording thread
    after it, so a slow save never holds the cancelled request."""
    import time

    recorder = _Recorder(seconds=1.0)
    save_started = threading.Event()
    release = threading.Event()

    def _slow_save(audio_data, transport):
        save_started.set()
        release.wait(timeout=5.0)
        return None

    with patch('voice_mode.tools.converse._save_cancelled_recording', new=_slow_save):
        start = time.monotonic()
        try:
            await _cancel_mid_recording_without_join(recorder)
        finally:
            elapsed = time.monotonic() - start
            release.set()
    assert elapsed < 2.0, f"cancel took {elapsed:.2f}s -- it waited on the save"
    assert save_started.wait(timeout=5.0)


async def _cancel_mid_recording_without_join(record_fake):
    from voice_mode.control_channel import get_control_state
    from voice_mode.tools.converse import listen_and_transcribe

    async def _noop_feedback(*_a, **_k):
        return None

    get_control_state().reset()
    try:
        with patch('voice_mode.tools.converse.play_audio_feedback', new=_noop_feedback), \
             patch('voice_mode.tools.converse.record_audio_with_silence_detection', new=record_fake):
            task = _start_listen(listen_and_transcribe, get_control_state())
            for _ in range(100):
                if record_fake.entered.is_set():
                    break
                await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        get_control_state().reset()
