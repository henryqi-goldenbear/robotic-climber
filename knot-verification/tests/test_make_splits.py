import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "knot-verification"))

from data_prep import make_splits


def test_load_annotations_falls_back_to_folder_labels(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    good_dir = images_dir / "kaggle_good"
    good_dir.mkdir(parents=True)
    bad_dir = images_dir / "manual_bad"
    bad_dir.mkdir(parents=True)
    (images_dir / "manual_good").mkdir()

    good_file = good_dir / "good_1.jpg"
    good_file2 = images_dir / "manual_good" / "good_2.jpg"
    bad_file = bad_dir / "bad_1.jpg"

    for path in [good_file, good_file2, bad_file]:
        Image.new("RGB", (10, 20), color="white").save(path)

    monkeypatch.setattr(make_splits.config, "RAW_IMAGES_DIR", images_dir)
    monkeypatch.setattr(make_splits.config, "ANNOTATIONS_CSV", tmp_path / "annotations.csv")

    df = make_splits.load_annotations()

    assert set(df["filename"]) == {
        "kaggle_good/good_1.jpg",
        "manual_good/good_2.jpg",
        "manual_bad/bad_1.jpg",
    }
    assert df["label"].tolist() == [1, 1, 0]
