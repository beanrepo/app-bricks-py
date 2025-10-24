# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import WaveGenerator, brick
from arduino.app_peripherals.speaker import Speaker
import threading
import numpy as np

from .effects import *
from .loaders import ABCNotationLoader


@brick
class SoundGenerator:
    SAMPLE_RATE = 16000
    A4_FREQUENCY = 440.0

    # Semitone mapping for the 12 notes (0 = C, 11 = B).
    # This is used to determine the relative position within an octave.
    SEMITONE_MAP = {
        "C": 0,
        "C#": 1,
        "DB": 1,
        "D": 2,
        "D#": 3,
        "EB": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "GB": 6,
        "G": 7,
        "G#": 8,
        "AB": 8,
        "A": 9,
        "A#": 10,
        "BB": 10,
        "B": 11,
    }

    NOTE_DURATTION = {
        "W": 1.0,  # Whole
        "H": 0.5,  # Half
        "Q": 0.25,  # Quarter
        "E": 0.125,  # Eighth
        "S": 0.0625,  # Sixteenth
        "T": 0.03125,  # Thirty-second
        "X": 0.015625,  # Sixty-fourth
    }

    # The reference point in the overall semitone count from C0. A4 is (4 * 12) + 9 semitones from C0.
    A4_SEMITONE_INDEX = (4 * 12) + 9

    def __init__(
        self,
        output_device: Speaker = None,
        bpm: int = 120,
        time_signature: tuple = (4, 4),
        octaves: int = 8,
        wave_form: str = "sine",
        master_volume: float = 1.0,
        sound_effects: list = None,
    ):
        """Initialize the SoundGenerator.
        Args:
            output_device (Speaker, optional): The output device to play sound through.
            wave_form (str): The type of wave form to generate. Supported values
                are "sine" (default), "square", "triangle" and "sawtooth".
            bpm (int): The tempo in beats per minute for note duration calculations.
            master_volume (float): The master volume level (0.0 to 1.0).
            octaves (int): Number of octaves to generate notes for (starting from octave
                0 up to octaves-1).
            sound_effects (list, optional): List of sound effect instances to apply to the audio
                signal (e.g., [SoundEffect.adsr()]). See SoundEffect class for available effects.
        """

        self._wave_gen = WaveGenerator(sample_rate=self.SAMPLE_RATE, wave_form=wave_form)
        self._bpm = bpm
        self.time_signature = time_signature
        self._master_volume = master_volume
        self._sound_effects = sound_effects
        if output_device is None:
            self._self_created_device = True
            self._output_device = Speaker(sample_rate=self.SAMPLE_RATE, format="FLOAT_LE")
        else:
            self._self_created_device = False
            self._output_device = output_device

        self._cfg_lock = threading.Lock()
        self._notes = {}
        for octave in range(octaves):
            notes = self._fill_node_frequencies(octave)
            self._notes.update(notes)

    def start(self):
        if self._self_created_device:
            self._output_device.start(notify_if_started=False)

    def stop(self):
        if self._self_created_device:
            self._output_device.stop()

    def set_master_volume(self, volume: float):
        """
        Set the master volume level.
        Args:
            volume (float): Volume level (0.0 to 1.0).
        """
        self._master_volume = max(0.0, min(1.0, volume))

    def set_effects(self, effects: list):
        """
        Set the list of sound effects to apply to the audio signal.
        Args:
            effects (list): List of sound effect instances (e.g., [SoundEffect.adsr()]).
        """
        with self._cfg_lock:
            self._sound_effects = effects

    def _fill_node_frequencies(self, octave: int) -> dict:
        """
        Given a sequence of notes with their names and octaves, fill in their frequencies.

        """
        notes = {}

        notes[f"REST"] = 0.0  # Rest note

        # Generate frequencies for all notes in the given octave
        for note_name in self.SEMITONE_MAP:
            frequency = self._note_to_frequency(note_name, octave)
            notes[f"{note_name}{octave}"] = frequency

        return notes

    def _note_to_frequency(self, note_name: str, octave: int) -> float:
        """
        Calculates the frequency (in Hz) of a musical note based on its name and octave.

        It uses the standard 12-tone equal temperament formula: f = f0 * 2^(n/12),
        where f0 is the reference frequency (A4=440Hz) and n is the number of
        semitones from the reference note.

        Args:
            note_name: The name of the note (e.g., 'A', 'C#', 'Bb', case-insensitive).
            octave: The octave number (e.g., 4 for A4, 5 for C5).

        Returns:
            The frequency in Hertz (float).
        """
        # 1. Normalize the note name for lookup
        normalized_note = note_name.strip().upper()
        if len(normalized_note) > 1 and normalized_note[1] == "#":
            # Ensure sharps are treated correctly (e.g., 'C#' is fine)
            pass
        elif len(normalized_note) > 1 and normalized_note[1].lower() == "b":
            # Replace 'B' (flat) with 'B' for consistent dictionary key
            normalized_note = normalized_note[0] + "B"

        # 2. Look up the semitone count within the octave
        if normalized_note not in self.SEMITONE_MAP:
            raise ValueError(f"Invalid note name: {note_name}. Please use notes like 'A', 'C#', 'Eb', etc.")

        semitones_in_octave = self.SEMITONE_MAP[normalized_note]

        # 3. Calculate the absolute semitone index (from C0)
        # Total semitones = (octave number * 12) + semitones_from_C_in_octave
        target_semitone_index = (octave * 12) + semitones_in_octave

        # 4. Calculate 'n', the number of semitones from the reference pitch (A4)
        # A4 is the reference, so n is the distance from A4.
        semitones_from_a4 = target_semitone_index - self.A4_SEMITONE_INDEX

        # 5. Calculate the frequency
        # f = 440 * 2^(n/12)
        frequency_hz = self.A4_FREQUENCY * (2.0 ** (semitones_from_a4 / 12.0))

        return frequency_hz

    def _note_duration(self, symbol: str | float | int) -> float:
        """
        Decode a note duration symbol into its corresponding fractional value.
        Args:
            symbol (str | float | int): Note duration symbol (e.g., 'W', 'H', 'Q', etc.) or a float/int value.
        Returns:
            float: Corresponding fractional duration value or the float itself if provided.
        """

        if isinstance(symbol, float) or isinstance(symbol, int):
            return self._compute_time_duration(symbol)

        duration = self.NOTE_DURATTION.get(symbol.upper(), None)
        if duration is not None:
            return self._compute_time_duration(duration)

        return self._compute_time_duration(1 / 4)  # Default to quarter note

    def _compute_time_duration(self, note_fraction: float) -> float:
        """
        Compute the time duration in seconds for a given note fraction and time signature.
        Args:
            note_fraction (float): The fraction of the note (e.g., 1.0 for whole, 0.5 for half).
            time_signature (tuple): The time signature as (numerator, denominator).
        Returns:
            float: Duration in seconds.
        """

        numerator, denominator = self.time_signature

        # For compound time signatures (6/8, 9/8, 12/8), the beat is the dotted quarter note (3/8)
        if denominator == 8 and numerator % 3 == 0:
            beat_value = 3 / 8
        else:
            beat_value = 1 / denominator  # es. 1/4 in 4/4

        # Calculate the duration of a single beat in seconds
        beat_duration = 60.0 / self._bpm

        # Compute the total duration
        return beat_duration * (note_fraction / beat_value)

    def _apply_sound_effects(self, signal: np.ndarray, frequency: float) -> np.ndarray:
        """
        Apply the configured sound effects to the audio signal.
        Args:
            signal (np.ndarray): Input audio signal.
        Returns:
            np.ndarray: Processed audio signal with sound effects applied.
        """
        with self._cfg_lock:
            if self._sound_effects is None:
                return signal

            processed_signal = signal
            for effect in self._sound_effects:
                if hasattr(effect, "apply_with_tone"):
                    processed_signal = effect.apply_with_tone(processed_signal, frequency)
                else:
                    processed_signal = effect.apply(processed_signal)

            return processed_signal

    def _get_note(self, note: str) -> float | None:
        if note is None:
            return None
        return self._notes.get(note.strip().upper())

    def play_chord(self, notes: list[str], note_duration: float | str = 1 / 4, volume: float = None):
        """
        Play a chord consisting of multiple musical notes simultaneously for a specified duration and volume.
        Args:
            notes (list[str]): List of musical notes to play (e.g., ['A4', 'C#5', 'E5']).
            note_duration (float | str): Duration of the chord as a float (like 1/4, 1/8) or a symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
        """
        duration = self._note_duration(note_duration)
        if len(notes) == 1:
            self.play(notes[0], duration, volume)
            return

        waves = []
        base_frequency = None
        for note in notes:
            frequency = self._get_note(note)
            if frequency:
                if base_frequency is None:
                    base_frequency = frequency
                if volume is None:
                    volume = self._master_volume
                data = self._wave_gen.generate_block(float(frequency), duration, volume)
                waves.append(data)
            else:
                continue
        if len(waves) == 0:
            return
        chord = np.sum(waves, axis=0, dtype=np.float32)
        chord /= np.max(np.abs(chord))  # Normalize to prevent clipping
        blk = chord.astype(np.float32)
        blk = self._apply_sound_effects(blk, base_frequency)
        try:
            self._output_device.play(blk, block_on_queue=False)
        except Exception as e:
            print(f"Error playing chord {notes}: {e}")

    def play(self, note: str, note_duration: float | str = 1 / 4, volume: float = None):
        """
        Play a musical note for a specified duration and volume.
        Args:
            note (str): The musical note to play (e.g., 'A4', 'C#5', 'REST').
            note_duration (float | str): Duration of the note as a float (like 1/4, 1/8) or a symbol ('W', 'H', 'Q', etc.).
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
        """
        duration = self._note_duration(note_duration)
        frequency = self._get_note(note)
        if frequency is not None and frequency >= 0.0:
            if volume is None:
                volume = self._master_volume
            data = self._wave_gen.generate_block(float(frequency), duration, volume)
            data = self._apply_sound_effects(data, frequency)
            self._output_device.play(data, block_on_queue=False)

    def play_tone(self, note: str, duration: float = 0.25, volume: float = None):
        """
        Play a musical note for a specified duration and volume.
        Args:
            note (str): The musical note to play (e.g., 'A4', 'C#5', 'REST').
            duration (float): Duration of the note as a float in seconds.
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
        """
        frequency = self._get_note(note)
        if frequency is not None and frequency >= 0.0 and duration > 0.0:
            if volume is None:
                volume = self._master_volume
            data = self._wave_gen.generate_block(float(frequency), duration, volume)
            data = self._apply_sound_effects(data, frequency)
            self._output_device.play(data, block_on_queue=False)

    def play_abc(self, abc_string: str, volume: float = None):
        """
        Play a sequence of musical notes defined in ABC notation.
        Args:
            abc_string (str): ABC notation string defining the sequence of notes.
            volume (float, optional): Volume level (0.0 to 1.0). If None, uses master volume.
        """
        if not abc_string or abc_string.strip() == "":
            return
        if volume is None:
            volume = self._master_volume
        metadata, notes = ABCNotationLoader.parse_abc_notation(abc_string)
        for note, duration in notes:
            frequency = self._get_note(note)
            if frequency is not None and frequency >= 0.0:
                data = self._wave_gen.generate_block(float(frequency), duration, volume)
                data = self._apply_sound_effects(data, frequency)
                self._output_device.play(data, block_on_queue=False)
