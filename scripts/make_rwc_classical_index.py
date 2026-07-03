import argparse
import csv
import hashlib
import json
import os

RWC_CLASSICAL_INDEX_PATH = "../mirdata/datasets/indexes/rwc_classical_index_2.0.json"


def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_rwc_classical_index(data_path):
    dataset_root = os.path.join(data_path, "rwc_classical")
    metadata_file = os.path.join(
        dataset_root,
        "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
        "metadata.csv",
    )

    with open(metadata_file, "r", encoding="utf-8") as fhandle:
        reader = csv.DictReader(fhandle, delimiter=";")
        track_ids = sorted(
            (row.get("RWCID") or "").strip()
            for row in reader
            if (row.get("CollID") or "").strip() == "C"
        )

    rwc_classical_index = {"version": "2.0", "tracks": {}, "metadata": {}}
    for track_id in track_ids:
        audio_rel = os.path.join("RWC-C", f"{track_id}.wav")
        audio_abs = os.path.join(dataset_root, audio_rel)
        if not os.path.exists(audio_abs):
            raise FileNotFoundError(f"Audio file not found: {audio_abs}")
        audio_rel = os.path.relpath(audio_abs, dataset_root).replace("\\", "/")
        audio_checksum = md5(audio_abs)

        beats_rel = os.path.join(
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "01_annotations_preprocessed",
            "beats",
            "RWC-C",
            f"{track_id}.csv",
        )
        beats_abs = os.path.join(dataset_root, beats_rel)
        if os.path.exists(beats_abs):
            beats_idx = beats_rel.replace("\\", "/")
            beats_md5 = md5(beats_abs)
        else:
            beats_idx, beats_md5 = None, None

        midi_rel = os.path.join(
            "rwc-annotations-2b84581b0c4c80514aadf7e9025a309c91e02cc2",
            "01_annotations_preprocessed",
            "MIDI_aligned",
            "RWC-C",
            f"{track_id}.mid",
        )
        midi_abs = os.path.join(dataset_root, midi_rel)
        if os.path.exists(midi_abs):
            midi_idx = midi_rel.replace("\\", "/")
            midi_md5 = md5(midi_abs)
        else:
            midi_idx, midi_md5 = None, None

        rwc_classical_index["tracks"][track_id] = {
            "audio": (audio_rel, audio_checksum),
            "beats": (beats_idx, beats_md5),
            "midi": (midi_idx, midi_md5),
        }

    metadata_rel = os.path.relpath(metadata_file, dataset_root).replace("\\", "/")
    rwc_classical_index["metadata"]["rwc-classical-metadata"] = (
        metadata_rel,
        md5(metadata_file),
    )

    with open(RWC_CLASSICAL_INDEX_PATH, "w", encoding="utf-8") as fhandle:
        json.dump(rwc_classical_index, fhandle, indent=2)

    print(f"Wrote {len(track_ids)} tracks to {RWC_CLASSICAL_INDEX_PATH}")


def main(args):
    make_rwc_classical_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make RWC-Classical index file.")
    PARSER.add_argument(
        "--data_path",
        type=str,
        help="Path to parent directory of rwc_classical.",
    )
    main(PARSER.parse_args())
