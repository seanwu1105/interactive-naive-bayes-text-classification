from interactive_naive_bayes.naive_bayes.classifier import train
from interactive_naive_bayes.naive_bayes.preprocessing import preprocess


def test_nbc():
    processed = preprocess()
    model = train(processed.categories, processed.documents)

    assert model.prior.shape == (8,)
    assert len(model.likelihood) == 8
