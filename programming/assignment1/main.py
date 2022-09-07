from interactive_naive_bayes.naive_bayes.classifier import predict, train
from interactive_naive_bayes.naive_bayes.preprocessing import preprocess, to_sample


def main():
    processed = preprocess()
    model = train(targets=processed.targets, samples=processed.samples)
    result = predict(
        sample=to_sample(
            text="architecture house complex material",
            label_indices=processed.label_indices,
        ),
        model=model,
    )

    print(processed.target_labels[result])


if __name__ == "__main__":
    main()
