"""
CLI: python run_on_image.py --image path/to/frame.jpg
"""
import argparse

from PIL import Image

from pipeline import KnotInspectionPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    pipeline = KnotInspectionPipeline()
    image = Image.open(args.image).convert("RGB")
    result = pipeline.predict(image)

    if result.label is None:
        print("No knot detected in frame.")
    else:
        print(f"Result: {result.label}  "
              f"(classifier confidence {result.confidence:.2%}, "
              f"detection confidence {result.detection_confidence:.2%})")


if __name__ == "__main__":
    main()
