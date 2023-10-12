# prep.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    print(f"Preprocessing the image: {args.image}")

if __name__ == "__main__":
    main()
