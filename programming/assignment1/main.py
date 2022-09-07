from interactive_naive_bayes.naive_bayes.classifier import train
from interactive_naive_bayes.naive_bayes.preprocessing import preprocess


def main():
    processed = preprocess()
    train(targets=processed.targets, samples=processed.samples)


if __name__ == "__main__":
    main()
