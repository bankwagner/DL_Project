# model_1.py and model_2.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()
    print(f"Running model with input: {args.input}")

if __name__ == "__main__":
    main()
