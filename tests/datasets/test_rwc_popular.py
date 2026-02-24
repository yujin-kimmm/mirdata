import os
import numpy as np

from mirdata.datasets import rwc_popular
from mirdata import annotations
from tests.test_utils import run_track_tests


def test_track():
    default_trackid = "RWC_P001"
    data_home = os.path.normpath("tests/resources/mir_datasets/rwc_popular")
    dataset = rwc_popular.Dataset(data_home, version="test")
    track = dataset.track(default_trackid)

    expected_attributes = {
        "track_id": "RWC_P001",
        "audio_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "RWC-P/1.wav",
        ),
        "sections_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "rwc-annotations-archive-main/AIST_RWC-MDB-P-2001_CHORUS/RM-P001.CHORUS.TXT",
        ),
        "beats_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "rwc-annotations-main/01_annotations_preprocessed/beats/RWC-P/RWC_P001.csv",
        ),
        "chords_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "rwc-annotations-main/01_annotations_preprocessed/chords/RWC-P/RWC_P001.csv",
        ),
        "voca_inst_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "rwc-annotations-archive-main/AIST_RWC-MDB-P-2001_VOCA_INST/RM-P001.VOCA_INST.TXT",
        ),
        "f0_path": os.path.join(
            os.path.normpath("tests/resources/mir_datasets/rwc_popular/"),
            "rwc-annotations-archive-main/AIST_RWC-MDB-P-2001_MELODY/RM-P001.MELODY.TXT",
        ),
        "piece_number": "1",
        "track_number": "1",
        "title": "Eien no replica",
        "artist": "Kazuo Nishi",
        "singer_information": "Male",
        "tempo": "135.0",
        "variation": "",
        "live_instrument": "Gt",
        "drum_information": "Drum sequences",
        "composer": "",
        "composition_type": "",
        "main_genre": "Popular",
        "sub_genre": "J-pop",
        "audio_start": "0.0236",
        "audio_end": "204.4778",
        "duration": "207.16829931972788",
    }

    expected_property_types = {
        "beats": annotations.BeatData,
        "sections": annotations.SectionData,
        "chords": annotations.ChordData,
        "vocal_instrument_activity": annotations.EventData,
        "melody": annotations.F0Data,
        "audio": tuple,
    }

    run_track_tests(track, expected_attributes, expected_property_types)

    # test audio loading functions
    y, sr = track.audio
    assert sr == 44100
    assert y.shape == (44100 * 2,)


def test_load_chords():
    chords_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "rwc-annotations-main/01_annotations_preprocessed/chords/RWC-P/RWC_P001.csv"
    )
    chord_data = rwc_popular.load_chords(chords_path)

    # check types
    assert type(chord_data) is annotations.ChordData
    assert type(chord_data.intervals) is np.ndarray
    assert type(chord_data.labels) is list

    # check values
    assert np.array_equal(chord_data.intervals[:3, 0], np.array([0.0, 0.104, 1.858]))
    assert np.array_equal(chord_data.intervals[:3, 1], np.array([0.104, 1.858, 3.646]))
    assert chord_data.labels[:3] == ["N", "Ab:min", "Gb:maj"]


def test_load_beats():
    beats_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "rwc-annotations-main/01_annotations_preprocessed/beats/RWC-P/RWC_P001.csv"
    )
    beat_data = rwc_popular.load_beats(beats_path)

    # check types
    assert type(beat_data) is annotations.BeatData
    assert type(beat_data.times) is np.ndarray
    assert type(beat_data.positions) is np.ndarray

    # check values
    assert np.array_equal(
        beat_data.times[:5], np.array([0.060, 0.510, 0.950, 1.390, 1.840])
    )
    assert np.array_equal(beat_data.positions[:5], np.array([1, 2, 3, 4, 1]))


def test_load_melody():
    melody_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "rwc-annotations-archive-main/AIST_RWC-MDB-P-2001_MELODY/RM-P001.MELODY.TXT"
    )
    melody_data = rwc_popular.load_melody(melody_path)

    # check types
    assert type(melody_data) is annotations.F0Data
    assert type(melody_data.times) is np.ndarray
    assert type(melody_data.frequencies) is np.ndarray
    assert type(melody_data.voicing) is np.ndarray

    # check values - should have uniform time intervals
    assert len(melody_data.times) > 0
    assert melody_data.times[0] == 10.18  # frame 1018 / 100
    assert melody_data.times[1] == 10.19  # frame 1019 / 100

    # check that times are uniform (difference of 0.01s for 10ms frames)
    time_diffs = np.diff(melody_data.times)
    assert np.allclose(time_diffs, 0.01, atol=1e-6)


def test_load_vocal_activity():
    vocinst_path = (
        "tests/resources/mir_datasets/rwc_popular/"
        + "rwc-annotations-archive-main/AIST_RWC-MDB-P-2001_VOCA_INST/RM-P001.VOCA_INST.TXT"
    )
    vocinst_data = rwc_popular.load_vocal_activity(vocinst_path)

    # check types
    assert type(vocinst_data) is annotations.EventData
    assert type(vocinst_data.intervals) is np.ndarray
    assert type(vocinst_data.events) is list

    # check values
    assert vocinst_data.intervals[0, 0] == 0.0
    assert vocinst_data.intervals[0, 1] == 10.293061224
    assert vocinst_data.events[0] == "b"


def test_load_metadata():
    data_home = "tests/resources/mir_datasets/rwc_popular"
    dataset = rwc_popular.Dataset(data_home, version="test")
    metadata = dataset._metadata

    assert "RWC_P001" in metadata
    assert metadata["RWC_P001"]["piece_number"] == "1"
    assert metadata["RWC_P001"]["track_number"] == "1"
    assert metadata["RWC_P001"]["title"] == "Eien no replica"
    assert metadata["RWC_P001"]["artist"] == "Kazuo Nishi"
    assert metadata["RWC_P001"]["singer_information"] == "Male"
    assert metadata["RWC_P001"]["tempo"] == "135.0"
    assert metadata["RWC_P001"]["live_instrument"] == "Gt"
    assert metadata["RWC_P001"]["drum_information"] == "Drum sequences"
    assert metadata["RWC_P001"]["main_genre"] == "Popular"
    assert metadata["RWC_P001"]["sub_genre"] == "J-pop"
    assert metadata["RWC_P001"]["duration"] == "207.16829931972788"
