"""RWC Classical Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    The RWC Classical Music Database is part of the RWC Music Database,
    consists of 50 classical pieces.

    This loader supports the RWC 2.0 release. RWC 2.0 re-releases the
    audio recordings and provides curated, preprocessed annotations with
    file names aligned to the released audio files. In the 2.0 file
    structure, the classical subset uses track identifiers such as
    ``RWC_C001`` and ``RWC_C025A``.

    For RWC-Classical, the curated preprocessed annotations currently
    exposed by this loader are:

    * Beat annotations
    * Aligned MIDI files

    **A note about the beat annotations:**

    Beat files are semicolon-separated CSV files with columns ``t`` and
    ``beat``. The ``t`` column stores beat times in seconds, and ``beat``
    stores the beat position/downbeat information.

    For more details, please visit: https://zenodo.org/records/18656623
    
"""

import csv
import os
from typing import Optional, Tuple

import numpy as np
import pretty_midi
from smart_open import open

from mirdata import annotations, core, download_utils, io

# these functions are identical for all rwc datasets
from mirdata.datasets.rwc_popular import (
    BIBTEX,
    DOWNLOAD_INFO,
    LICENSE_INFO,
    load_audio,
    load_beats,
)

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="rwc_classical_index_2.0.json",
        url="https://zenodo.org/records/21180405/files/rwc_classical_index_2.0.json?download=1",
        checksum="8b4112b936d91bb737f41cfc17a880e8",
    ),
    "sample": core.Index(filename="rwc_classical_index_2.0_sample.json"),
}

REMOTES = {
    "audio": download_utils.RemoteFileMetadata(
        filename="RWC-C.zip",
        url="https://zenodo.org/records/18656623/files/RWC-C.zip?download=1",
        checksum="2ac9139c4f03a65885ae0d0d299f67f8",
    ),
    "annotation": download_utils.RemoteFileMetadata(
        filename="rwc-annotations-main.zip",
        url="https://github.com/rwc-music/rwc-annotations/archive/2b84581b0c4c80514aadf7e9025a309c91e02cc2.zip",
        checksum="40f17835554467f66de60602f72f5686",
    ),
}

class Track(core.Track):
    """rwc_classical Track class

    Args:
        track_id (str): track id of the track

    Attributes:
        artist (str): the track's artist
        audio_path (str): path of the audio file
        beats_path (str): path of the beat annotation file
        category (str): One of 'Symphony', 'Concerto', 'Orchestral',
            'Solo', 'Chamber', 'Vocal', or blank.
        composer (str): Composer of this Track.
        duration (float): Duration of the track in seconds
        piece_number (str): Piece number of this Track, [1-50]
        sections_path (str): path of the section annotation file
        suffix (str): string within M01-M06
        title (str): Title of The track.
        track_id (str): track id
        track_number (str): CD track number of this Track

    Cached Properties:
        sections (SectionData): human-labeled section annotations
        beats (BeatData): human-labeled beat annotations

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):
        super().__init__(track_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.beats_path = self.get_path("beats")
        self.midi_path = self.get_path("midi")

    @property
    def piece_number(self):
        return self._track_metadata.get("piece_number")

    @property
    def cd_number(self):
        return self._track_metadata.get("cd_number")

    @property
    def track_number(self):
        return self._track_metadata.get("track_number")

    @property
    def title(self):
        return self._track_metadata.get("title")

    @property
    def composer(self):
        return self._track_metadata.get("composer")

    @property
    def artist(self):
        return self._track_metadata.get("artist")

    @property
    def live_instrument(self):
        return self._track_metadata.get("live_instrument")

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
    def beats(self) -> Optional[annotations.BeatData]:
        return load_beats(self.beats_path)

    @core.cached_property
    def midi(self) -> Optional[pretty_midi.PrettyMIDI]:
        return io.load_midi(self.midi_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The rwc_classical dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rwc_classical",
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
            self.data_home,
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "metadata.csv",
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
            if line["CollID"] == "C":
                track_id = line["RWCID"]
                metadata_index[track_id] = {
                    "piece_number": line["PieceNo"],
                    "cd_number": line["CDNo"],
                    "track_number": line["TrackNo"],
                    "title": line["Title"],
                    "artist": line["Artist"],
                    "live_instrument": line["LiveInstruments"],
                    "composer": line["Composer"],
                    "composition_type": line["CompositionType"],
                    "main_genre": line["GenreMain"],
                    "sub_genre": line["GenreSub"],
                    "audio_start": line["audio_start"],
                    "audio_end": line["audio_end"],
                    "duration": line["duration"],
                }

        return metadata_index
