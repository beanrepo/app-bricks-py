# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_bricks.sound_generator import ABCNotationLoader


def test_abc_loader():
    full_abc = """
    X:1
    T:Main Theme
    M:4/4
    L:1/8
    Q:1/4=130
    K:Cm
    "Cm"E2 E2 E2 "Ab"C>G | "Cm"E2 "Ab"C>G "Cm"E4 |
    "Cm"B2 B2 B2 "Ab"c>G | "Fm"^D#2 "Ab"C>G "Cm"E4 |
    """

    reference_notes = [
        ("E4", 60 / 130),
        ("E4", 60 / 130),
        ("E4", 60 / 130),
        ("C4", (60 / 130) / 2),
        ("G4", (60 / 130) / 2),
        ("E4", 60 / 130),
        ("C4", (60 / 130) / 2),
        ("G4", (60 / 130) / 2),
        ("E4", (60 / 130) * 2),
        ("B4", 60 / 130),
        ("B4", 60 / 130),
        ("B4", 60 / 130),
        ("C5", (60 / 130) / 2),
        ("G4", (60 / 130) / 2),
        ("D#4", 60 / 130),
        ("C4", (60 / 130) / 2),
        ("G4", (60 / 130) / 2),
        ("E4", (60 / 130) * 2),
    ]

    metadata, loaded = ABCNotationLoader.parse_abc_notation(full_abc)
    assert metadata["title"] == "Main Theme"
    assert "transpose" not in metadata
    assert metadata["tempo"] == "1/4=130"

    i_ref = 0
    for note, duration in loaded:
        print(f"Note: {note}, Duration: {duration}")
        assert note == reference_notes[i_ref][0]
        assert abs(duration - reference_notes[i_ref][1]) < 0.01
        i_ref += 1


def test_abc_loader_with_transpose():
    full_abc = """
    X:1
    T:Main Theme
    M:4/4
    L:1/8
    Q:1/4=130
    K:Cm
    %%transpose -12
    "Cm"E2 E2 E2 "Ab"C>G | "Cm"E2 "Ab"C>G "Cm"E4 |
    "Cm"B2 B2 B2 "Ab"c>G | "Fm"^D#2 "Ab"C>G "Cm"E4 |
    """

    reference_notes = [
        ("E3", 60 / 130),
        ("E3", 60 / 130),
        ("E3", 60 / 130),
        ("C3", (60 / 130) / 2),
        ("G3", (60 / 130) / 2),
        ("E3", 60 / 130),
        ("C3", (60 / 130) / 2),
        ("G3", (60 / 130) / 2),
        ("E3", (60 / 130) * 2),
        ("B3", 60 / 130),
        ("B3", 60 / 130),
        ("B3", 60 / 130),
        ("C4", (60 / 130) / 2),
        ("G3", (60 / 130) / 2),
        ("D#3", 60 / 130),
        ("C3", (60 / 130) / 2),
        ("G3", (60 / 130) / 2),
        ("E3", (60 / 130) * 2),
    ]

    metadata, loaded = ABCNotationLoader.parse_abc_notation(full_abc)
    assert metadata["title"] == "Main Theme"
    assert "transpose" in metadata
    assert metadata["transpose"] == -1
    assert metadata["tempo"] == "1/4=130"

    i_ref = 0
    for note, duration in loaded:
        print(f"Note: {note}, Duration: {duration}")
        assert note == reference_notes[i_ref][0]
        assert abs(duration - reference_notes[i_ref][1]) < 0.01
        i_ref += 1
