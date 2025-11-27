# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import numpy as np
import threading
import time
from arduino.app_bricks.wave_generator import WaveGenerator
import arduino.app_utils.app as app
from arduino.app_utils import AppController


@pytest.fixture
def app_instance(monkeypatch):
    """Provides a fresh AppController instance for each test."""
    instance = AppController()
    monkeypatch.setattr(app, "App", instance)
    return instance


@pytest.fixture(autouse=True)
def mock_speaker(monkeypatch):
    """Mock Speaker to avoid hardware dependencies."""

    class FakeSpeaker:
        # Class attributes (macros like real Speaker)
        USB_SPEAKER_1 = "USB_SPEAKER_1"
        USB_SPEAKER_2 = "USB_SPEAKER_2"

        def __init__(self, device=None, sample_rate=16000, channels=1, format="FLOAT_LE", periodsize=None, queue_maxsize=100):
            self.device = device or "fake_device"
            self.sample_rate = sample_rate
            self.channels = channels
            self.format = format
            self._periodsize = periodsize  # Support new periodsize parameter
            self._is_started = False
            self._is_reproducing = threading.Event()  # Support lifecycle checks
            self._played_data = []
            self._mixer = FakeMixer()

        def start(self):
            self._is_started = True
            self._is_reproducing.set()

        def stop(self):
            self._is_started = False
            self._is_reproducing.clear()

        def play(self, data, block_on_queue=False):
            if self._is_started:
                self._played_data.append(data)

        def set_volume(self, volume: int):
            self._mixer.setvolume(volume)

        def is_started(self):
            return self._is_started

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()
            return False

    class FakeMixer:
        def __init__(self):
            self._volume = 100

        def setvolume(self, volume: int):
            self._volume = max(0, min(100, volume))

        def getvolume(self):
            return [self._volume]

    # Patch Speaker in the wave_generator module
    monkeypatch.setattr("arduino.app_bricks.wave_generator.wave_generator.Speaker", FakeSpeaker)
    return FakeSpeaker


def test_wave_generator_initialization_default(mock_speaker):
    """Test WaveGenerator initializes with default parameters."""
    wave_gen = WaveGenerator()

    assert wave_gen.sample_rate == 16000
    assert wave_gen.wave_type == "sine"
    assert wave_gen.block_duration == 0.03
    assert wave_gen.attack == 0.01
    assert wave_gen.release == 0.03
    assert wave_gen.glide == 0.02
    assert wave_gen._speaker is not None
    assert wave_gen._speaker.sample_rate == 16000


def test_wave_generator_initialization_custom(mock_speaker):
    """Test WaveGenerator initializes with custom parameters."""
    wave_gen = WaveGenerator(
        sample_rate=48000,
        wave_type="square",
        block_duration=0.05,
        attack=0.02,
        release=0.05,
        glide=0.03,
    )

    assert wave_gen.sample_rate == 48000
    assert wave_gen.wave_type == "square"
    assert wave_gen.block_duration == 0.05
    assert wave_gen.attack == 0.02
    assert wave_gen.release == 0.05
    assert wave_gen.glide == 0.03
    assert wave_gen._speaker.sample_rate == 48000


def test_wave_generator_with_external_speaker(mock_speaker):
    """Test WaveGenerator with externally provided Speaker."""
    external_speaker = mock_speaker(device="external_device", sample_rate=16000)
    wave_gen = WaveGenerator(speaker=external_speaker)

    assert wave_gen._speaker is external_speaker
    assert wave_gen._speaker.device == "external_device"


def test_wave_generator_start_stop(app_instance, mock_speaker):
    """Test WaveGenerator start and stop methods."""
    wave_gen = WaveGenerator()

    # Initially not running
    assert not wave_gen._running.is_set()

    # Start the generator
    wave_gen.start()
    assert wave_gen._running.is_set()
    assert wave_gen._speaker.is_started()
    assert wave_gen._producer_thread is not None
    assert wave_gen._producer_thread.is_alive()

    time.sleep(0.1)  # Let it run briefly

    # Stop the generator
    wave_gen.stop()
    assert not wave_gen._running.is_set()
    assert not wave_gen._speaker.is_started()

    # Wait for thread to finish
    time.sleep(0.1)


def test_set_frequency(mock_speaker):
    """Test setting frequency."""
    wave_gen = WaveGenerator()

    wave_gen.set_frequency(440.0)
    assert wave_gen._target_freq == 440.0

    wave_gen.set_frequency(880.0)
    assert wave_gen._target_freq == 880.0

    # Test negative frequency (should be clamped to 0)
    wave_gen.set_frequency(-100.0)
    assert wave_gen._target_freq == 0.0


def test_set_amplitude(mock_speaker):
    """Test setting amplitude."""
    wave_gen = WaveGenerator()

    wave_gen.set_amplitude(0.5)
    assert wave_gen._target_amp == 0.5

    wave_gen.set_amplitude(1.0)
    assert wave_gen._target_amp == 1.0

    # Test out of range (should be clamped)
    wave_gen.set_amplitude(1.5)
    assert wave_gen._target_amp == 1.0

    wave_gen.set_amplitude(-0.5)
    assert wave_gen._target_amp == 0.0


def test_set_wave_type(mock_speaker):
    """Test setting wave type."""
    wave_gen = WaveGenerator()

    wave_gen.set_wave_type("sine")
    assert wave_gen.wave_type == "sine"

    wave_gen.set_wave_type("square")
    assert wave_gen.wave_type == "square"

    wave_gen.set_wave_type("sawtooth")
    assert wave_gen.wave_type == "sawtooth"

    wave_gen.set_wave_type("triangle")
    assert wave_gen.wave_type == "triangle"

    # Test invalid wave type
    with pytest.raises(ValueError):
        wave_gen.set_wave_type("invalid")


def test_set_volume(mock_speaker):
    """Test setting hardware volume."""
    wave_gen = WaveGenerator()

    wave_gen.set_volume(70)
    assert wave_gen._speaker._mixer._volume == 70

    wave_gen.set_volume(100)
    assert wave_gen._speaker._mixer._volume == 100

    # Test get_volume
    assert wave_gen.get_volume() == 100

    wave_gen.set_volume(50)
    assert wave_gen.get_volume() == 50


def test_set_envelope_params(mock_speaker):
    """Test setting envelope parameters."""
    wave_gen = WaveGenerator()

    wave_gen.set_envelope_params(attack=0.05)
    assert wave_gen.attack == 0.05

    wave_gen.set_envelope_params(release=0.1)
    assert wave_gen.release == 0.1

    wave_gen.set_envelope_params(glide=0.04)
    assert wave_gen.glide == 0.04

    # Test all at once
    wave_gen.set_envelope_params(attack=0.02, release=0.06, glide=0.03)
    assert wave_gen.attack == 0.02
    assert wave_gen.release == 0.06
    assert wave_gen.glide == 0.03

    # Test negative values (should be clamped to 0)
    wave_gen.set_envelope_params(attack=-0.01)
    assert wave_gen.attack == 0.0


def test_get_state(mock_speaker):
    """Test getting current generator state."""
    wave_gen = WaveGenerator()

    wave_gen.set_frequency(440.0)
    wave_gen.set_amplitude(0.8)
    wave_gen.set_wave_type("square")
    wave_gen.set_volume(90)

    state = wave_gen.get_state()

    assert "frequency" in state
    assert "amplitude" in state
    assert "wave_type" in state
    assert state["wave_type"] == "square"
    assert "volume" in state
    assert state["volume"] == 90
    assert "phase" in state


def test_generate_block_sine(mock_speaker):
    """Test generating a sine wave block."""
    wave_gen = WaveGenerator(sample_rate=16000)

    # Generate a block
    block = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sine")

    # Check block properties
    assert isinstance(block, np.ndarray)
    assert block.dtype == np.float32
    expected_samples = int(16000 * 0.03)  # block_duration = 0.03
    assert len(block) == expected_samples
    # Check amplitude is within range
    assert np.max(np.abs(block)) <= 0.5


def test_generate_block_square(mock_speaker):
    """Test generating a square wave block."""
    wave_gen = WaveGenerator(sample_rate=16000)

    block = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="square")

    assert isinstance(block, np.ndarray)
    # Square wave has envelope applied, so check amplitude range
    assert np.max(np.abs(block)) <= 0.5


def test_generate_block_sawtooth(mock_speaker):
    """Test generating a sawtooth wave block."""
    wave_gen = WaveGenerator(sample_rate=16000)

    _ = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sawtooth")

    # Verify internal state updated correctly
    assert wave_gen._buf_samples is not None


def test_generate_block_triangle(mock_speaker):
    """Test generating a triangle wave block."""
    wave_gen = WaveGenerator(sample_rate=16000)

    _ = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="triangle")

    # Verify internal state updated correctly
    assert wave_gen._buf_samples is not None


def test_frequency_glide(mock_speaker):
    """Test frequency glide (portamento) effect."""
    wave_gen = WaveGenerator(sample_rate=16000, glide=0.1)

    # Set initial frequency
    wave_gen._current_freq = 220.0

    # Generate block with new target frequency
    _ = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sine")

    # Current frequency should have moved towards target but not reached it
    # (because glide time is longer than block duration)
    assert wave_gen._current_freq > 220.0
    assert wave_gen._current_freq < 440.0


def test_amplitude_envelope(mock_speaker):
    """Test amplitude envelope (attack/release)."""
    wave_gen = WaveGenerator(sample_rate=16000, attack=0.1, release=0.1)

    # Set initial amplitude
    wave_gen._current_amp = 0.0

    # Generate block with new target amplitude
    _ = wave_gen._generate_block(freq_target=440.0, amp_target=0.8, wave_type="sine")

    # Current amplitude should have moved towards target but not reached it
    assert wave_gen._current_amp > 0.0
    assert wave_gen._current_amp < 0.8


def test_producer_loop_generates_audio(app_instance, mock_speaker):
    """Test that producer loop generates and plays audio."""
    wave_gen = WaveGenerator()

    wave_gen.set_frequency(440.0)
    wave_gen.set_amplitude(0.5)
    wave_gen.start()

    # Let it run for a bit
    time.sleep(0.2)

    # Check that audio was played
    assert len(wave_gen._speaker._played_data) > 0

    wave_gen.stop()


def test_thread_safety(mock_speaker):
    """Test thread-safe access to parameters."""
    wave_gen = WaveGenerator()

    def set_params():
        for i in range(100):
            wave_gen.set_frequency(440.0 + i)
            wave_gen.set_amplitude(0.5)
            time.sleep(0.001)

    def get_state():
        for i in range(100):
            state = wave_gen.get_state()
            assert "frequency" in state
            time.sleep(0.001)

    wave_gen.start()

    # Start multiple threads accessing the generator
    threads = [
        threading.Thread(target=set_params),
        threading.Thread(target=get_state),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=5)

    wave_gen.stop()


def test_buffer_preallocation(mock_speaker):
    """Test that buffers are pre-allocated and reused."""
    wave_gen = WaveGenerator(sample_rate=16000)

    # Generate first block
    block1 = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sine")

    # Check buffers are allocated
    assert wave_gen._buf_N > 0
    assert wave_gen._buf_phase_incs is not None
    assert wave_gen._buf_phases is not None
    assert wave_gen._buf_envelope is not None
    assert wave_gen._buf_samples is not None

    # Generate second block
    _ = wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sine")

    # Buffers should still be the same size (reused)
    assert wave_gen._buf_N == len(block1)


def test_phase_continuity(mock_speaker):
    """Test that phase is continuous across blocks."""
    wave_gen = WaveGenerator(sample_rate=16000)

    initial_phase = wave_gen._phase

    # Generate multiple blocks
    for _ in range(10):
        wave_gen._generate_block(freq_target=440.0, amp_target=0.5, wave_type="sine")

    # Phase should have advanced
    assert wave_gen._phase != initial_phase
    # Phase should be wrapped to [0, 2π]
    assert 0.0 <= wave_gen._phase < 2 * np.pi


def test_zero_amplitude_produces_silence(mock_speaker):
    """Test that zero amplitude produces silent output."""
    wave_gen = WaveGenerator(sample_rate=16000)

    block = wave_gen._generate_block(freq_target=440.0, amp_target=0.0, wave_type="sine")

    # All samples should be zero or very close to zero
    assert np.allclose(block, 0.0, atol=1e-6)


def test_app_controller_integration(app_instance, mock_speaker):
    """Test integration with AppController (start/stop via App)."""
    wave_gen = WaveGenerator()

    # Register manually to avoid auto-registration
    app_instance.unregister(wave_gen)
    app_instance.start_brick(wave_gen)

    assert wave_gen._running.is_set()
    assert wave_gen._speaker.is_started()

    time.sleep(0.1)

    app_instance.stop_brick(wave_gen)

    assert not wave_gen._running.is_set()
    assert not wave_gen._speaker.is_started()


def test_multiple_start_stop_cycles(app_instance, mock_speaker):
    """Test starting and stopping multiple times."""
    wave_gen = WaveGenerator()

    for _ in range(3):
        wave_gen.start()
        assert wave_gen._running.is_set()
        time.sleep(0.05)

        wave_gen.stop()
        assert not wave_gen._running.is_set()
        time.sleep(0.05)


def test_double_start_warning(app_instance, mock_speaker):
    """Test that starting an already running generator logs a warning."""
    wave_gen = WaveGenerator()

    wave_gen.start()
    assert wave_gen._running.is_set()

    # Try to start again (should warn but not crash)
    wave_gen.start()
    assert wave_gen._running.is_set()

    wave_gen.stop()


def test_double_stop_warning(app_instance, mock_speaker):
    """Test that stopping a non-running generator logs a warning."""
    wave_gen = WaveGenerator()

    # Try to stop before starting (should warn but not crash)
    wave_gen.stop()
    assert not wave_gen._running.is_set()
