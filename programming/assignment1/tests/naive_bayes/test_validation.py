from interactive_naive_bayes.naive_bayes.preprocessing import preprocess
from interactive_naive_bayes.naive_bayes.validation import validate


def test_10_folds():
    processed = preprocess()
    best_model, _ = validate(10, processed.categories, processed.documents)

    assert best_model.prior.shape == (8,)
    assert len(best_model.likelihood) == 8
