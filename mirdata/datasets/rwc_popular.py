"""RWC Popular Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The Popular Music Database consists of 100 songs — 20 songs with English lyrics
    performed in the style of popular music typical of songs on the American hit
    charts in the 1980s, and 80 songs with Japanese lyrics performed in the style of
    modern Japanese popular music typical of songs on the Japanese hit charts in
    the 1990s.

    For more details, please visit: https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-p.html

"""

import csv
import os
from typing import Optional, TextIO, Tuple

import numpy as np
from smart_open import open

from mirdata import annotations, core, download_utils, io

# these functions are identical for all rwc datasets
from mirdata.datasets.rwc_classical import (
    load_sections,
    load_audio,
    LICENSE_INFO,
)

BIBTEX = """@inproceedings{GotoHNO02_RWC_ISMIR,
  address = {Paris, France},
  author = {Masataka Goto and Hiroki Hashiguchi and Takuichi Nishimura and Ryuichi Oka},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  pages = {287--288},
  title = {{RWC} Music Database: Popular, Classical and Jazz Music Databases},
  year = {2002}
}
@inproceedings{GotoHNO03_RWCGenre_ISMIR,
  author    = {Masataka Goto and Hiroki Hashiguchi and Takuichi Nishimura and Ryuichi Oka},
  title     = {{RWC} Music Database: Music genre database and musical instrument sound database},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  address   = {Baltimore, Maryland, USA},
  year      = {2003},
  pages     = {229--230},
}
@inproceedings{Goto06_AnnotationsAIST_ISMIR,
  author    = {Masataka Goto},
  title     = {{AIST} Annotation for the {RWC} Music Database},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  year      = {2006},
  pages     = {359--360},
}"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="rwc_popular_index_2.0.json",
        url="https://zenodo.org/records/18751784/files/rwc_popular_index_2.0.json?download=1",
        checksum="9c09e1efe56173f9a3ad513671817054",
    ),
    "sample": core.Index(filename="rwc_popular_index_2.0_sample.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="RWC-P.zip.zip",
        url="https://zenodo.org/records/18656623/files/RWC-P.zip?download=1",
        checksum="960a11a2d7fb603ad0dae8428f53d4f0",
    ),
    "annotation": download_utils.RemoteFileMetadata(
        filename="rwc-annotations-main.zip",
        url="https://github.com/rwc-music/rwc-annotations/archive/refs/heads/main.zip",
        checksum="83d518e2f905d9fbabb993f4ea54f1cd",
    ),
    "annotation_archive": download_utils.RemoteFileMetadata(
        filename="rwc-annotations-archive-main.zip",
        url="https://github.com/rwc-music/rwc-annotations-archive/archive/refs/heads/main.zip",
        checksum="330354fb59a261e0704c64f7b2e8794c",
    ),
}

DOWNLOAD_INFO = """
This dataset is RWC Music Database version 2.0 (released 2026).

If you need the previous version 1.0 with its loader and annotations, install an older mirdata version:
  pip install "mirdata<=1.0.0"
"""


class Track(core.Track):
    """rwc_popular Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path of the audio file
        sections_path (str): path of the section annotation file
        beats_path (str): path of the beat annotation file
        chords_path (str): path of the chord annotation file
        voca_inst_path (str): path of the vocal/instrumental annotation file
        f0_path (str): path of the melody (F0) annotation file
        piece_number (str): Piece number
        track_number (str): CD track number
        title (str): title of the track
        artist (str): artist name
        singer_information (str): singer information (male, female, or vocal group)
        tempo (str): tempo of the track in BPM
        variation (str): variation information
        live_instrument (str): live instrument information
        drum_information (str): drum information (e.g., 'Drum sequences', 'Live drums', 'Drum loops')
        composer (str): composer name
        composition_type (str): type of composition
        main_genre (str): main genre of the track
        sub_genre (str): sub genre of the track
        audio_start (str): audio start time
        audio_end (str): audio end time
        duration (str): duration of the track in seconds

    Cached Properties:
        sections (SectionData): human-labeled section annotation
        beats (BeatData): human-labeled beat annotation
        chords (ChordData): human-labeled chord annotation
        vocal_instrument_activity (EventData): human-labeled vocal/instrument activity annotation
        melody (F0Data): human-labeled melody (F0) annotation

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.sections_path = self.get_path("sections")
        self.beats_path = self.get_path("beats")
        self.chords_path = self.get_path("chords")
        self.voca_inst_path = self.get_path("voca_inst")
        self.f0_path = self.get_path("melody")

    @property
    def piece_number(self):
        return self._track_metadata.get("piece_number")

    @property
    def track_number(self):
        return self._track_metadata.get("track_number")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def singer_information(self):
        return self._track_metadata.get("singer_information")

    @property
    def tempo(self):
        return self._track_metadata.get("tempo")

    @property
    def variation(self):
        return self._track_metadata.get("variation")

    @property
    def live_instrument(self):
        return self._track_metadata.get("live_instrument")

    @property
    def drum_information(self):
        return self._track_metadata.get("drum_information")

    @property
    def composer(self):
        return self._track_metadata.get("composer")

    @property
    def composition_type(self):
        return self._track_metadata.get("composition_type")

    @property
    def main_genre(self):
        return self._track_metadata.get("main_genre")

    @property
    def sub_genre(self):
        return self._track_metadata.get("sub_genre")

    @property
    def audio_start(self):
        return self._track_metadata.get("audio_start")

    @property
    def audio_end(self):
        return self._track_metadata.get("audio_end")

    @property
    def duration(self):
        return self._track_metadata.get("duration")

    @core.cached_property
    def sections(self) -> Optional[annotations.SectionData]:
        return load_sections(self.sections_path)

    @core.cached_property
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def chords(self) -> Optional[annotations.ChordData]:
        return load_chords(self.chords_path)

    @core.cached_property
    def vocal_instrument_activity(self) -> Optional[annotations.EventData]:
        return load_vocal_activity(self.voca_inst_path)

    @core.cached_property
    def melody(self) -> Optional[annotations.F0Data]:
        return load_melody(self.f0_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> annotations.ChordData:
    """Load rwc chord data from a file

    Args:
        fhandle (str or file-like): File-like object or path to chord annotation file

    Returns:
        ChordData: chord data

    """
    begs = []  # timestamps of chord beginnings
    ends = []  # timestamps of chord endings
    chords = []  # chord labels

    reader = csv.reader(fhandle, delimiter=";")
    next(reader, None)  # skip header
    for row in reader:
        begs.append(float(row[0]))
        ends.append(float(row[1]))
        chords.append(row[2])

    return annotations.ChordData(np.array([begs, ends]).T, "s", chords, "open")


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> annotations.BeatData:
    """Load rwc beat data from a file

    Args:
        fhandle (str or file-like): File-like object or path to beats annotation file

    Returns:
        BeatData: beat data

    """

    beat_times = []  # timestamps of beat interval beginnings
    beat_positions = []  # beat position inside the bar

    reader = csv.reader(fhandle, delimiter=";")
    next(reader, None)
    for row in reader:
        beat_times.append(float(row[0]))
        beat_positions.append(int(row[1]))

    return annotations.BeatData(
        np.array(beat_times), "s", np.array(beat_positions).astype(int), "bar_index"
    )


@io.coerce_to_string_io
def load_melody(fhandle: TextIO) -> Optional[annotations.F0Data]:
    """Load rwc_popular f0 annotations

    Args:
        fhandle (str or file-like): path or file-like object pointing
            to melody annotation file

    Returns:
        F0Data: predominant melody

    """
    frame_data = {}
    for row in csv.reader(fhandle, delimiter="\t"):
        frame_num = int(row[0])
        freq = float(row[3])
        frame_data[frame_num] = freq

    min_frame = min(frame_data.keys())
    max_frame = max(frame_data.keys())

    times = []
    freqs = []
    voicing = []

    for frame in range(min_frame, max_frame + 1):
        times.append(frame / 100.0)
        freq = frame_data.get(frame, 0.0)  # return 0 for empty frame
        freqs.append(freq)
        voicing.append(1.0 if freq > 0 else 0.0)

    return annotations.F0Data(
        np.array(times), "s", np.array(freqs), "hz", np.array(voicing), "binary"
    )


@io.coerce_to_string_io
def load_vocal_activity(fhandle: TextIO) -> annotations.EventData:
    """Load rwc vocal activity data from a file

    Args:
        fhandle (str or file-like): File-like object or path to vocal activity annotation file

    Returns:
        EventData: vocal activity data

    """
    begs = []  # timestamps of vocal-instrument activity beginnings
    ends = []  # timestamps of vocal-instrument activity endings
    events = []  # vocal-instrument activity labels

    reader = csv.reader(fhandle, delimiter="\t")
    raw_data = []
    for line in reader:
        if line[0] != "Piece No.":
            raw_data.append(line)

    for i in range(len(raw_data)):
        # Parsing vocal-instrument activity as intervals (beg, end, event)
        if raw_data[i] != raw_data[-1]:
            begs.append(float(raw_data[i][0]))
            ends.append(float(raw_data[i + 1][0]))
            events.append(raw_data[i][1])

    return annotations.EventData(np.array([begs, ends]).T, "s", events, "open")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_popular dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_popular",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(
            self.data_home, "rwc-annotations-main", "metadata.csv"
        )
        try:
            with open(metadata_path, "r", encoding="utf-8") as fhandle:
                reader = csv.DictReader(fhandle, delimiter=";")
                raw_data = []
                for row in reader:
                    raw_data.append(row)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}
        for line in raw_data:
            if line["CollID"] == "P":
                track_id = line["RWCID"]
                metadata_index[track_id] = {
                    "piece_number": line["PieceNo"],
                    "cd_number": line["CDNo"],
                    "track_number": line["TrackNo"],
                    "title": line["Title"],
                    "artist": line["Artist"],
                    "singer_information": line["SingerInformation"],
                    "singing_language": line["SingingLanguage"],
                    "tempo": line["Tempo"],
                    "variation": line["Variation"],
                    "live_instrument": line["LiveInstruments"],
                    "drum_information": line["DrumInformation"],
                    "composer": line["Composer"],
                    "composition_type": line["CompositionType"],
                    "main_genre": line["GenreMain"],
                    "sub_genre": line["GenreSub"],
                    "audio_start": line["audio_start"],
                    "audio_end": line["audio_end"],
                    "duration": line["duration"],
                }
            else:
                pass

        return metadata_index
