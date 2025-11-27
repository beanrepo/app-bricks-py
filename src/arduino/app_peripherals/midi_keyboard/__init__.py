# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import logging
import threading
import time
from typing import Callable, Optional, Dict
from arduino.app_utils import Logger

logger = Logger("MIDIKeyboard", logging.DEBUG)


class MidiMessage:
    """Minimal MIDI message container (compatible with mido.Message interface)."""

    def __init__(self, msg_type: str, **kwargs):
        self.type = msg_type
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != "type")
        return f"<MidiMessage {self.type} {attrs}>"


class MIDIKeyboardException(Exception):
    """Custom exception for MIDI keyboard errors."""

    pass


class MIDIKeyboard:
    """MIDI keyboard/controller input peripheral for Linux/ALSA.

    Handles MIDI input from USB MIDI devices including keyboards, drum pads,
    and control surfaces. Provides callbacks for note events, control changes,
    and pitch bend.

    Uses direct ALSA device access (/dev/snd/midiC*D*) without external dependencies.
    Designed for Arduino UNO Q and other Linux boards with ALSA support.
    """

    USB_MIDI_1 = "USB_MIDI_1"
    USB_MIDI_2 = "USB_MIDI_2"

    def __init__(
        self,
        device: Optional[str] = USB_MIDI_1,
        channel: Optional[int] = None,
        profile: Optional[str] = None,
    ):
        """Initialize the MIDI keyboard.

        Args:
            device (str, optional): MIDI device name or USB_MIDI_1/2 macro.
                If None, uses the first available MIDI input.
            channel (int, optional): MIDI channel to filter (1-16).
                If None, receives from all channels.
            profile (str, optional): Controller profile name for semantic mapping.
                If None, uses generic/raw MIDI mode.

        Raises:
            MIDIKeyboardException: If no MIDI device is found or cannot be opened.
        """
        logger.info(
            "Init MIDIKeyboard with device=%s, channel=%s, profile=%s",
            device,
            channel,
            profile,
        )

        self.channel = channel
        self.profile_name = profile
        self._port = None
        self._is_running = threading.Event()
        self._listener_thread = None

        # Callback registries
        self._note_on_callbacks: Dict[int, Callable] = {}
        self._note_off_callbacks: Dict[int, Callable] = {}
        self._cc_callbacks: Dict[int, Callable] = {}
        self._pitchbend_callback: Optional[Callable] = None
        self._aftertouch_callback: Optional[Callable] = None

        # Global callbacks (all notes/cc)
        self._global_note_on_callback: Optional[Callable] = None
        self._global_note_off_callback: Optional[Callable] = None
        self._global_cc_callback: Optional[Callable] = None

        # Semantic callbacks (when using profiles)
        self._semantic_pad_callbacks: Dict[str, Callable] = {}
        self._semantic_knob_callbacks: Dict[str, Callable] = {}

        # Load profile if specified
        self._profile = None
        if profile:
            from .profiles import load_profile

            self._profile = load_profile(profile)
            logger.info(f"Loaded profile: {self._profile.name}")

        # Resolve device
        self.device_name = self._resolve_device(device)
        logger.info(f"Using MIDI device: {self.device_name}")

    def _open_alsa_seq(self, device_name: str):
        """Open ALSA sequencer port directly when mido backends are unavailable.

        Args:
            device_name: Device name in hw:X,Y format

        Returns:
            AlsaSeqPort object with receive() method compatible with mido

        Raises:
            Exception: If ALSA device cannot be opened
        """
        import select

        class AlsaSeqPort:
            """Minimal ALSA sequencer port wrapper compatible with mido interface."""

            def __init__(self, device_name):
                # Convert hw:X,Y to /dev/snd/midiCXDY
                if device_name.startswith("hw:"):
                    parts = device_name[3:].split(",")
                    card = int(parts[0])
                    dev = int(parts[1]) if len(parts) > 1 else 0
                    self.device_path = f"/dev/snd/midiC{card}D{dev}"
                else:
                    self.device_path = device_name

                self.fd = open(self.device_path, "rb", buffering=0)
                logger.info(f"Opened raw ALSA device: {self.device_path}")

            def receive(self, block=True):
                """Read MIDI message from device (compatible with mido interface)."""
                if not block:
                    # Non-blocking: check if data is available
                    rlist, _, _ = select.select([self.fd], [], [], 0)
                    if not rlist:
                        return None

                # Read MIDI message (variable length)
                try:
                    status_byte = self.fd.read(1)
                    if not status_byte:
                        return None

                    status = status_byte[0]

                    # Parse message based on status byte
                    if (status & 0xF0) == 0x90:  # Note On
                        data = self.fd.read(2)
                        note, velocity = data[0], data[1]
                        return MidiMessage("note_on", note=note, velocity=velocity, channel=status & 0x0F)

                    elif (status & 0xF0) == 0x80:  # Note Off
                        data = self.fd.read(2)
                        note, velocity = data[0], data[1]
                        return MidiMessage("note_off", note=note, velocity=velocity, channel=status & 0x0F)

                    elif (status & 0xF0) == 0xB0:  # Control Change
                        data = self.fd.read(2)
                        control, value = data[0], data[1]
                        return MidiMessage("control_change", control=control, value=value, channel=status & 0x0F)

                    elif (status & 0xF0) == 0xE0:  # Pitch Bend
                        data = self.fd.read(2)
                        lsb, msb = data[0], data[1]
                        pitch = ((msb << 7) | lsb) - 8192
                        return MidiMessage("pitchwheel", pitch=pitch, channel=status & 0x0F)

                    elif (status & 0xF0) == 0xD0:  # Channel Pressure (Aftertouch)
                        data = self.fd.read(1)
                        value = data[0]
                        return MidiMessage("aftertouch", value=value, channel=status & 0x0F)

                    elif (status & 0xF0) == 0xA0:  # Polyphonic Aftertouch
                        data = self.fd.read(2)
                        note, value = data[0], data[1]
                        return MidiMessage("polytouch", note=note, value=value, channel=status & 0x0F)

                    else:
                        # Unknown or system message, skip
                        logger.debug(f"Skipping MIDI status byte: 0x{status:02x}")
                        return None

                except Exception as e:
                    logger.error(f"Error parsing ALSA MIDI message: {e}")
                    return None

            def close(self):
                """Close the ALSA device."""
                if self.fd:
                    self.fd.close()
                    self.fd = None

        return AlsaSeqPort(device_name)

    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve MIDI device name, handling USB_MIDI_1/2 macros.

        Args:
            device: Device name or USB_MIDI_1/2 macro

        Returns:
            Resolved MIDI device name

        Raises:
            MIDIKeyboardException: If no MIDI device is found
        """
        available = self.list_usb_devices()

        if not available:
            raise MIDIKeyboardException("No MIDI input devices found.")

        if device is None or device == self.USB_MIDI_1:
            return available[0]

        if device == self.USB_MIDI_2:
            if len(available) < 2:
                raise MIDIKeyboardException(f"USB_MIDI_2 requested but only {len(available)} device(s) found.")
            return available[1]

        # Check if device exists in available list
        if device in available:
            return device

        # Try partial match
        for dev in available:
            if device.lower() in dev.lower():
                logger.info(f"Matched device '{device}' to '{dev}'")
                return dev

        raise MIDIKeyboardException(f"MIDI device '{device}' not found. Available: {available}")

    @staticmethod
    def list_usb_devices() -> list:
        """List available MIDI input devices (ALSA raw MIDI).

        Returns:
            List of MIDI input device names in hw:X,Y format
        """
        import glob
        import re

        # Scan /dev/snd for ALSA raw MIDI devices
        devices = []
        midi_devices = glob.glob("/dev/snd/midiC*D*")

        if midi_devices:
            # Found raw MIDI devices, convert to hw:X,Y format
            for dev in sorted(midi_devices):
                # Extract card and device numbers
                # Format: /dev/snd/midiC0D0 -> hw:0,0
                match = re.search(r"midiC(\d+)D(\d+)", dev)
                if match:
                    card, device = match.groups()
                    hw_name = f"hw:{card},{device}"
                    devices.append(hw_name)

            logger.info(f"Available MIDI inputs (ALSA): {devices}")
            return devices
        else:
            logger.warning("No MIDI devices found in /dev/snd")
            return []

    def start(self):
        """Open MIDI port and start listening for messages."""
        if self._is_running.is_set():
            logger.warning("MIDIKeyboard is already running")
            return

        # Open ALSA sequencer directly (no external dependencies)
        try:
            self._port = self._open_alsa_seq(self.device_name)
            logger.info(f"Opened ALSA sequencer port: {self.device_name}")
        except Exception as e:
            raise MIDIKeyboardException(f"Failed to open MIDI device: {e}")

        self._is_running.set()
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True, name="MIDIKeyboard-Listener")
        self._listener_thread.start()
        logger.info("MIDIKeyboard started")

    def stop(self):
        """Stop listening and close MIDI port."""
        if not self._is_running.is_set():
            logger.warning("MIDIKeyboard is not running")
            return

        logger.info("Stopping MIDIKeyboard...")
        self._is_running.clear()

        if self._listener_thread:
            self._listener_thread.join(timeout=2)
            if self._listener_thread.is_alive():
                logger.warning("Listener thread did not terminate in time")
            self._listener_thread = None

        if self._port:
            try:
                self._port.close()
                logger.info("MIDI port closed")
            except Exception as e:
                logger.warning(f"Error closing MIDI port: {e}")
            self._port = None

        logger.info("MIDIKeyboard stopped")

    def is_connected(self) -> bool:
        """Check if MIDI device is still connected and running.

        Returns:
            True if device is connected and listening thread is alive
        """
        return self._is_running.is_set() and self._listener_thread and self._listener_thread.is_alive()

    def _listen_loop(self):
        """Main listener loop running in background thread."""
        logger.debug("MIDI listener loop started")

        while self._is_running.is_set():
            try:
                # Non-blocking receive with timeout
                msg = self._port.receive(block=False)

                if msg is None:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue

                # Filter by channel if specified
                if self.channel is not None and hasattr(msg, "channel"):
                    if msg.channel + 1 != self.channel:  # mido uses 0-based channels
                        continue

                self._process_message(msg)

            except (OSError, IOError) as e:
                # Device disconnected
                logger.error(f"MIDI device disconnected: {e}")
                self._is_running.clear()
                break
            except Exception as e:
                logger.error(f"Error in MIDI listener loop: {e}")
                time.sleep(0.1)

        logger.debug("MIDI listener loop terminated")

    def _process_message(self, msg):
        """Process incoming MIDI message and trigger callbacks.

        Args:
            msg: MidiMessage or mido.Message object
        """
        try:
            if msg.type == "note_on":
                note = msg.note
                velocity = msg.velocity

                # Treat note_on with velocity=0 as note_off
                if velocity == 0:
                    self._handle_note_off(note, 0)
                else:
                    self._handle_note_on(note, velocity)

            elif msg.type == "note_off":
                self._handle_note_off(msg.note, msg.velocity)

            elif msg.type == "control_change":
                self._handle_cc(msg.control, msg.value)

            elif msg.type == "pitchwheel":
                self._handle_pitchbend(msg.pitch)

            elif msg.type == "aftertouch":
                self._handle_aftertouch(msg.value)

            elif msg.type == "polytouch":
                # Polyphonic aftertouch (per-note pressure)
                logger.debug(f"Poly aftertouch: note={msg.note}, value={msg.value}")

        except Exception as e:
            logger.error(f"Error processing MIDI message {msg}: {e}")

    def _handle_note_on(self, note: int, velocity: int):
        """Handle note on event."""
        logger.debug(f"Note ON: {note}, velocity: {velocity}")

        # Specific note callback
        if note in self._note_on_callbacks:
            try:
                self._note_on_callbacks[note](velocity)
            except Exception as e:
                logger.error(f"Error in note_on callback for note {note}: {e}")

        # Global note on callback
        if self._global_note_on_callback:
            try:
                self._global_note_on_callback(note, velocity)
            except Exception as e:
                logger.error(f"Error in global note_on callback: {e}")

        # Semantic callback (profile-based)
        if self._profile and note in self._profile.note_map:
            semantic_name = self._profile.note_map[note]
            if semantic_name in self._semantic_pad_callbacks:
                try:
                    self._semantic_pad_callbacks[semantic_name](velocity)
                except Exception as e:
                    logger.error(f"Error in semantic pad callback for {semantic_name}: {e}")

    def _handle_note_off(self, note: int, velocity: int):
        """Handle note off event."""
        logger.debug(f"Note OFF: {note}, velocity: {velocity}")

        # Specific note callback
        if note in self._note_off_callbacks:
            try:
                self._note_off_callbacks[note](velocity)
            except Exception as e:
                logger.error(f"Error in note_off callback for note {note}: {e}")

        # Global note off callback
        if self._global_note_off_callback:
            try:
                self._global_note_off_callback(note, velocity)
            except Exception as e:
                logger.error(f"Error in global note_off callback: {e}")

    def _handle_cc(self, control: int, value: int):
        """Handle control change event."""
        logger.debug(f"CC: {control}, value: {value}")

        # Specific CC callback
        if control in self._cc_callbacks:
            try:
                self._cc_callbacks[control](value)
            except Exception as e:
                logger.error(f"Error in CC callback for control {control}: {e}")

        # Global CC callback
        if self._global_cc_callback:
            try:
                self._global_cc_callback(control, value)
            except Exception as e:
                logger.error(f"Error in global CC callback: {e}")

        # Semantic callback (profile-based)
        if self._profile and control in self._profile.cc_map:
            semantic_name = self._profile.cc_map[control]
            if semantic_name in self._semantic_knob_callbacks:
                try:
                    self._semantic_knob_callbacks[semantic_name](value)
                except Exception as e:
                    logger.error(f"Error in semantic knob callback for {semantic_name}: {e}")

    def _handle_pitchbend(self, value: int):
        """Handle pitch bend event."""
        logger.debug(f"Pitch bend: {value}")

        if self._pitchbend_callback:
            try:
                self._pitchbend_callback(value)
            except Exception as e:
                logger.error(f"Error in pitchbend callback: {e}")

    def _handle_aftertouch(self, value: int):
        """Handle aftertouch (channel pressure) event."""
        logger.debug(f"Aftertouch: {value}")

        if self._aftertouch_callback:
            try:
                self._aftertouch_callback(value)
            except Exception as e:
                logger.error(f"Error in aftertouch callback: {e}")

    # Callback registration methods

    def on_note_on(self, callback: Callable, note: Optional[int] = None):
        """Register callback for note on events.

        Args:
            callback: Callback function(note, velocity) or function(velocity) if note specified
            note: Specific note number (0-127). If None, callback receives all note on events.

        Example:
            # All notes
            midi.on_note_on(lambda note, vel: print(f"Note {note}: {vel}"))

            # Specific note
            midi.on_note_on(lambda vel: print(f"C4: {vel}"), note=60)
        """
        if note is None:
            self._global_note_on_callback = callback
        else:
            self._note_on_callbacks[note] = callback

    def on_note_off(self, callback: Callable, note: Optional[int] = None):
        """Register callback for note off events.

        Args:
            callback: Callback function(note, velocity) or function(velocity) if note specified
            note: Specific note number (0-127). If None, callback receives all note off events.
        """
        if note is None:
            self._global_note_off_callback = callback
        else:
            self._note_off_callbacks[note] = callback

    def on_control_change(self, callback: Callable, control: Optional[int] = None):
        """Register callback for control change events.

        Args:
            callback: Callback function(control, value) or function(value) if control specified
            control: Specific CC number (0-127). If None, callback receives all CC events.

        Example:
            # All CCs
            midi.on_control_change(lambda cc, val: print(f"CC {cc}: {val}"))

            # Specific CC
            midi.on_control_change(lambda val: print(f"Modwheel: {val}"), control=1)
        """
        if control is None:
            self._global_cc_callback = callback
        else:
            self._cc_callbacks[control] = callback

    def on_pitch_bend(self, callback: Callable):
        """Register callback for pitch bend events.

        Args:
            callback: Callback function(value) where value is -8192 to +8191

        Example:
            midi.on_pitch_bend(lambda val: print(f"Pitch bend: {val}"))
        """
        self._pitchbend_callback = callback

    def on_aftertouch(self, callback: Callable):
        """Register callback for aftertouch (channel pressure) events.

        Args:
            callback: Callback function(value) where value is 0-127
        """
        self._aftertouch_callback = callback

    def on_pad(self, pad_name: str, callback: Callable):
        """Register callback for semantic pad name (requires profile).

        Args:
            pad_name: Semantic pad name from profile (e.g., "pad_1", "kick")
            callback: Callback function(velocity)

        Raises:
            MIDIKeyboardException: If no profile is loaded

        Example:
            midi = MIDIKeyboard(profile="akai_mpc_mini")
            midi.on_pad("pad_1", lambda vel: print(f"Pad 1: {vel}"))
        """
        if not self._profile:
            raise MIDIKeyboardException("on_pad() requires a profile. Initialize with profile parameter.")
        self._semantic_pad_callbacks[pad_name] = callback

    def on_knob(self, knob_name: str, callback: Callable):
        """Register callback for semantic knob name (requires profile).

        Args:
            knob_name: Semantic knob name from profile (e.g., "knob_1", "filter_cutoff")
            callback: Callback function(value)

        Raises:
            MIDIKeyboardException: If no profile is loaded

        Example:
            midi = MIDIKeyboard(profile="akai_mpk_mini_plus")
            midi.on_knob("knob_1", lambda val: print(f"Knob 1: {val}"))
        """
        if not self._profile:
            raise MIDIKeyboardException("on_knob() requires a profile. Initialize with profile parameter.")
        self._semantic_knob_callbacks[knob_name] = callback

    @staticmethod
    def note_to_frequency(note: int) -> float:
        """Convert MIDI note number to frequency in Hz.

        Args:
            note: MIDI note number (0-127)

        Returns:
            Frequency in Hz

        Example:
            freq = MIDIKeyboard.note_to_frequency(69)  # A4 = 440 Hz
        """
        return 440.0 * (2.0 ** ((note - 69) / 12.0))

    @staticmethod
    def frequency_to_note(frequency: float) -> int:
        """Convert frequency in Hz to nearest MIDI note number.

        Args:
            frequency: Frequency in Hz

        Returns:
            MIDI note number (0-127)
        """
        import math

        return int(round(69 + 12 * math.log2(frequency / 440.0)))

    def get_profile_info(self) -> Optional[dict]:
        """Get current profile information.

        Returns:
            Profile info dict or None if no profile loaded
        """
        if not self._profile:
            return None

        return {
            "name": self._profile.name,
            "note_map": self._profile.note_map,
            "cc_map": self._profile.cc_map,
            "has_aftertouch": self._profile.has_aftertouch,
            "has_pitchbend": self._profile.has_pitchbend,
        }

    @staticmethod
    def list_profiles() -> list:
        """List available controller profiles.

        Returns:
            List of profile names
        """
        from .profiles import list_available_profiles

        return list_available_profiles()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.stop()
        return False

    def __del__(self):
        """Ensure MIDI port is closed when object is destroyed."""
        try:
            self.stop()
        except Exception:
            pass
