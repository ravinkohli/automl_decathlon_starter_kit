from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _approximate_mode, check_random_state
from sklearn.utils.validation import _num_samples, check_array
from typing import List
import warnings


class CustomStratifiedShuffleSplit(StratifiedShuffleSplit):
    """Splitter that deals with classes with too few samples"""

    def _iter_indices(self, X, y, groups=None):  # type: ignore
        n_samples = _num_samples(X)
        y = check_array(y, ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            default_test_size=self._default_test_size,
        )

        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)

        if n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes)
            )
        if n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes)
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"), np.cumsum(class_counts)[:-1]
        )

        rng = check_random_state(self.random_state)

        for _ in range(self.n_splits):
            # if there are ties in the class-counts, we want
            # to make sure to break them anew in each iteration
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)
            train = []
            test = []


            for i, class_count in enumerate(n_i):
                if class_count == 0:
                    t_i[i] -= 1
                    n_i[i] += 1

                    j = np.argmax(n_i)
                    if n_i[j] == 1:
                        warnings.warn(
                            "Can't respect size requirements for split.",
                            " The training set must contain all of the unique"
                            " labels that exist in the dataset.",
                        )
                    else:
                        n_i[j] -= 1
                        t_i[j] += 1

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                train.extend(perm_indices_class_i[: n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]: n_i[i] + t_i[i]])

            train = rng.permutation(train)
            test = rng.permutation(test)

            yield train, test


def subsample(
    X,
    is_classification,
    sample_size,
    y, 
    random_state, 
): 

    if isinstance(X, List):
        X = np.asarray(X)
    if isinstance(y, List):
        y = np.asarray(y)

    if is_classification and y is not None:
        splitter = CustomStratifiedShuffleSplit(
            train_size=sample_size, random_state=random_state
        )
        indices_to_keep, _ = next(splitter.split(X=X, y=y))
        X, y = _subsample_by_indices(X, y, indices_to_keep)

    elif y is None:
        X, _ = train_test_split(  # type: ignore
            X,
            train_size=sample_size,
            random_state=random_state,
        )
    else:
        X, _, y, _ = train_test_split(  # type: ignore
            X,
            y,
            train_size=sample_size,
            random_state=random_state,
        )

    return X, y


def _subsample_by_indices(
    X, 
    y, 
    indices_to_keep, 
): 
    if ispandas(X):
        idxs = X.index[indices_to_keep]
        X = X.loc[idxs]
    else:
        X = X[indices_to_keep]

    if ispandas(y):
        # Ifnoring types as mypy does not infer y as dataframe.
        idxs = y.index[indices_to_keep]  # type: ignore [index]
        y = y.loc[idxs]  # type: ignore [union-attr]
    else:
        y = y[indices_to_keep]
    return X, y

def ispandas(X) -> bool:
    """ Whether X is pandas.DataFrame or pandas.Series """
    return hasattr(X, "iloc")