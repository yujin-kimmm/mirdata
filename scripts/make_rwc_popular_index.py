import argparse
import csv
import hashlib
import json
import os

RWC_POPULAR_INDEX_PATH = "../mirdata/datasets/indexes/rwc_popular_index_2.0.json"


def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as fhandle:
        for chunk in iter(lambda: fhandle.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def make_rwc_popular_index(data_path):
    dataset_root = os.path.join(data_path, "rwc_popular")
    metadata_file = os.path.join(dataset_root, "rwc-annotations-main", "metadata.csv")

    with open(metadata_file, "r", encoding="utf-8") as fhandle:
        reader = csv.DictReader(fhandle, delimiter=";")
        piece = []
        for row in reader:
            rwcid = (row.get("RWCID") or "").strip()
            if not rwcid.startswith("RWC_P"):
                continue

            piece.append(rwcid[-3:])

    track_ids = sorted([f"RM-P{p}" for p in piece])

    rwc_popular_index = {"version": "2.0", "tracks": {}, "metadata": {}}
    for track_id in track_ids:
        pid = track_id[-3:]
        rwcid = f"RWC_P{pid}"

        audio_rel = os.path.join("RWC-P", f"{rwcid}.wav")
        audio_abs = os.path.join(dataset_root, audio_rel)
        if not os.path.exists(audio_abs):
            raise FileNotFoundError(f"Audio file not found: {audio_abs}")
        audio_rel = os.path.relpath(audio_abs, dataset_root).replace("\\", "/")
        audio_checksum = md5(audio_abs)

        beats_rel = os.path.join(
            "rwc-annotations-main",
            "01_annotations_preprocessed",
            "beats",
            "RWC-P",
            f"{rwcid}.csv",
        )
        beats_abs = os.path.join(dataset_root, beats_rel)
        if os.path.exists(beats_abs):
            beats_idx = beats_rel.replace("\\", "/")
            beats_md5 = md5(beats_abs)
        else:
            beats_idx, beats_md5 = None, None

        chords_rel = os.path.join(
            "rwc-annotations-main",
            "01_annotations_preprocessed",
            "chords",
            "RWC-P",
            f"{rwcid}.csv",
        )
        chords_abs = os.path.join(dataset_root, chords_rel)
        if os.path.exists(chords_abs):
            chords_idx = chords_rel.replace("\\", "/")
            chords_md5 = md5(chords_abs)
        else:
            chords_idx, chords_md5 = None, None

        sections_rel = os.path.join(
            "rwc-annotations-archive-main",
            "AIST_RWC-MDB-P-2001_CHORUS",
            f"{track_id}.CHORUS.TXT",
        )
        sections_abs = os.path.join(dataset_root, sections_rel)
        if os.path.exists(sections_abs):
            sections_idx = sections_rel.replace("\\", "/")
            sections_md5 = md5(sections_abs)
        else:
            sections_idx, sections_md5 = None, None

        voca_inst_rel = os.path.join(
            "rwc-annotations-archive-main",
            "AIST_RWC-MDB-P-2001_VOCA_INST",
            f"{track_id}.VOCA_INST.TXT",
        )
        voca_inst_abs = os.path.join(dataset_root, voca_inst_rel)
        if os.path.exists(voca_inst_abs):
            voca_inst_idx = voca_inst_rel.replace("\\", "/")
            voca_inst_md5 = md5(voca_inst_abs)
        else:
            voca_inst_idx, voca_inst_md5 = None, None

        melody_rel = os.path.join(
            "rwc-annotations-archive-main",
            "AIST_RWC-MDB-P-2001_MELODY",
            f"{track_id}.MELODY.TXT",
        )

        melody_abs = os.path.join(dataset_root, melody_rel)
        if os.path.exists(melody_abs):
            melody_idx = melody_rel.replace("\\", "/")
            melody_md5 = md5(melody_abs)
        else:
            melody_idx, melody_md5 = None, None

        rwc_popular_index["tracks"][track_id] = {
            "audio": (audio_rel, audio_checksum),
            "sections": (sections_idx, sections_md5),
            "beats": (beats_idx, beats_md5),
            "chords": (chords_idx, chords_md5),
            "voca_inst": (voca_inst_idx, voca_inst_md5),
            "melody": (melody_idx, melody_md5),
        }

    metadata_rel = os.path.relpath(metadata_file, dataset_root).replace("\\", "/")
    rwc_popular_index["metadata"]["rwc-popular-metadata"] = (
        metadata_rel,
        md5(metadata_file),
    )

    with open(RWC_POPULAR_INDEX_PATH, "w", encoding="utf-8") as fhandle:
        json.dump(rwc_popular_index, fhandle, indent=2)

    print(f"Wrote {len(track_ids)} tracks to {RWC_POPULAR_INDEX_PATH}")


def main(args):
    make_rwc_popular_index(args.rwc_popular_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make RWC-Popular index file.")
    PARSER.add_argument(
        "rwc_popular_data_path",
        type=str,
        help="Path to parent directory of rwc_popular_new.",
    )
    main(PARSER.parse_args())
