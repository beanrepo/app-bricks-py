# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils.logger import Logger
from typing import List, Tuple
import re

logger = Logger(__name__)


class ABCNotationLoader:
    NOT_HANDLED_RESERVED_LINES = r"^[A-Za-z]:"

    @staticmethod
    def _parse_abc_duration(duration_str: str, default_duration_in_seconds: float) -> float:
        """
        Parse ABC duration notation (e.g., '2', '/2', '3/2').
        The returned duration is in absolute seconds.
        - default_duration_in_seconds: The absolute duration (in seconds) of the
          note specified by the L: field, calculated using the Q: field (BPM).
        Args:
            duration_str (str): Duration string from ABC notation.
            default_duration_in_seconds (float): Default duration in seconds for a single unit.
        Returns:
            float: Calculated duration in seconds.
        """
        if not duration_str:
            return default_duration_in_seconds

        # Handle fractions (e.g., C/2, C/4)
        if "/" in duration_str:
            parts = duration_str.split("/")
            # Handles C/ (division by 2) and C4/2 (multiplication by 4, division by 2)
            numerator = int(parts[0]) if parts[0] else 1
            denominator = int(parts[1]) if len(parts) > 1 and parts[1] else 2

            return default_duration_in_seconds * numerator / denominator

        try:
            multiplier = int(duration_str)
            return default_duration_in_seconds * multiplier
        except ValueError:
            return default_duration_in_seconds

    @staticmethod
    def parse_abc_notation(abc_string: str, default_octave: int = 4) -> Tuple[dict, List[Tuple[str, float]]]:
        """
        Parse ABC notation and convert to an array of (note, duration_in_seconds) tuples.
        Args:
            abc_string (str): ABC notation string.
            default_octave (int): Default octave for uppercase notes (C4).

        Returns:
            Tuple[dict, List[Tuple[str, float]]]: Metadata dictionary and list of (note, duration) tuples.
        """

        metadata = {}

        lines = abc_string.split("\n")
        music_lines = []

        # --- Parse Header Fields ---
        for line in lines:
            line = line.strip()
            if re.match(ABCNotationLoader.NOT_HANDLED_RESERVED_LINES, line):
                if line.startswith("X:"):
                    metadata["reference"] = line[2:].strip()
                elif line.startswith("T:"):
                    metadata["title"] = line[2:].strip()
                elif line.startswith("K:"):
                    metadata["key"] = line[2:].strip()
                elif line.startswith("L:"):
                    metadata["default_length"] = line[2:].strip()
                elif line.startswith("Q:"):
                    metadata["tempo"] = line[2:].strip()
                elif line.startswith("M:"):
                    metadata["meter"] = line[2:].strip()
                elif line.startswith("C:"):
                    metadata["composer"] = line[2:].strip()
                elif line.startswith("R:"):
                    metadata["rhythm"] = line[2:].strip()
            elif not line.startswith("%") and line:
                music_lines.append(line)

        # Standard ABC default for L: is 1/8 if not specified.
        default_unit_fraction = 1 / 8

        if "default_length" in metadata and metadata["default_length"]:
            match_L = re.match(r"(\d+)/(\d+)", metadata["default_length"])
            if match_L:
                num, denom = int(match_L.group(1)), int(match_L.group(2))
                default_unit_fraction = num / denom

        bpm = 120  # Default BPM if Q: is not specified
        beat_unit_fraction = 0.25  # Default beat unit (1/4 or quarter note)

        if "tempo" in metadata and metadata["tempo"]:
            # Q: field is typically 'note_fraction=BPM', e.g. '1/4=120'
            match_Q = re.match(r"(\d+/\d+)=(\d+)", metadata["tempo"].replace(" ", ""))

            if match_Q:
                note_str, bpm_str = match_Q.groups()
                bpm = int(bpm_str)

                q_num, q_denom = map(int, note_str.split("/"))
                beat_unit_fraction = q_num / q_denom
            else:
                try:
                    bpm = int(metadata["tempo"].replace(" ", ""))
                except ValueError:
                    pass  # Keep default BPM

        # Duration in seconds of the note specified as the beat unit (Q: note)
        duration_of_beat_unit = 60.0 / bpm

        # Calculate the ratio between the default L: unit and the Q: beat unit.
        # This handles cases where L: and Q: define different note values (e.g., L:1/16, Q:1/4=120)
        ratio_to_beat_unit = default_unit_fraction / beat_unit_fraction

        # The absolute duration in seconds of the note defined by L:
        default_duration_in_seconds = ratio_to_beat_unit * duration_of_beat_unit

        # Informational output
        if "title" in metadata:
            logger.info(f"Playing: {metadata['title']}")
        logger.info(f"BPM: {bpm}, Beat Unit Fraction: {beat_unit_fraction:.3f}, Default L: {default_unit_fraction:.3f}")
        logger.info(f"Duration of 1 beat: {duration_of_beat_unit:.3f}s. Default L: Duration: {default_duration_in_seconds:.3f}s")

        # --- 5. Parse Music Lines ---
        music_string = " ".join(music_lines)
        result = []

        # Tokenize notes, rests, and bar lines
        music_string = re.sub(r'"[^"]*"', "", music_string)  # Remove chord annotations
        tokens = re.findall(r"[A-Ga-g][',]*[#b]?[0-9]*/?[0-9]*|z[0-9]*/?[0-9]*|\|", music_string)

        for token in tokens:
            if token == "|":
                continue

            # Parse Rest
            if token.startswith("z"):
                # Use the duration in seconds as the base unit
                duration = ABCNotationLoader._parse_abc_duration(token[1:], default_duration_in_seconds)
                result.append(("REST", duration))
                continue

            # Parse Note
            note_char = token[0]
            rest = token[1:]

            octave = default_octave
            if note_char.islower():
                octave = octave + 1
                note_char = note_char.upper()

            # Handle octave markers (',) - adjust octave accordingly - increase/decrease octave
            octave_markers = re.findall(r"[',]", rest)
            for marker in octave_markers:
                if marker == "'":
                    octave += 1
                elif marker == ",":
                    octave -= 1

            rest = re.sub(r"[',]", "", rest)

            # Handle accidentals (# sharp, b flat)
            accidental = ""
            if rest and rest[0] in ["#", "b"]:
                accidental = rest[0].upper()
                rest = rest[1:]

            duration = ABCNotationLoader._parse_abc_duration(rest, default_duration_in_seconds)

            # Build note name (e.g., C#4)
            note_name = f"{note_char}{accidental}{octave}"
            result.append((note_name, duration))

        metadata["actual_bpm"] = bpm
        return metadata, result
