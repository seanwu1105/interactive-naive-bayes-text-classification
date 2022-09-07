from interactive_naive_bayes.naive_bayes.classifier import predict, train
from interactive_naive_bayes.naive_bayes.preprocessing import preprocess, to_sample


def test_naive_bayes():
    processed = preprocess()
    model = train(targets=processed.targets, samples=processed.samples)
    result = predict(
        sample=to_sample(
            text="architecture house complex material",
            label_indices=processed.label_indices,
        ),
        model=model,
    )

    assert processed.target_labels[result] == "Building"
