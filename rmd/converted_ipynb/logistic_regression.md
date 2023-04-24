## Logistic Regression

We would like to perform hyperparameter selection over the set of all logistic regression models.

Logistic regression with $K > 2$ classes is actually broken into two "flavours":
- **one-versus-all**: Fit a separate binary classifier for each class against the rest, and classify according to the highest score.
- **multinomial**: Fit a single classifier with $K$ outputs (one of them is 1), and take the softmax thereof. This is the flavour we learn in STAT 441, and is the default used by `sklearn`.

We choose to focus on multinomial logistic regression here.

### Setup

```python
from model import SKLogisticRegression, DuplicateSKLogisticRegression
from tune import (
    Constant,
    outer_cv,
    Real,
    SKBayesTuner,
)
```

### Search Space

For regularization, ElasticNet encompasses both L1 and L2 penalties (and with a weak enough regularization term, no-penalty as well).
We thus parameterize the search space by the ratio of the L1 and L2 penalties, and the coefficient C representing the extent of regularization.

Unfortunately, the only solver in `sklearn` that works with ElasticNet is the `saga` solver, and that solver is quite sluggish.

```python
space = dict(
    # constants
    multi_class=Constant("multinomial"),
    penalty=Constant("elasticnet"),
    solver=Constant("saga"),
    max_iter=Constant(9001),  # practically unlimited
    random_state=Constant(441),
    warm_start=Constant(True),  # try to speed up optimization
    
    # variables
    C=Real(low=1e-4, high=1e4, prior="log-uniform"),
    l1_ratio=Real(low=0.0, high=1.0, prior="uniform"),
)
```

### Base

```python
outer_cv(
    tuner=SKBayesTuner(SKLogisticRegression()),
    search=space,
    name="Logistic Regression",
    duplicate=False,
    hard_mode=False,
)
```

### Hard

```python
outer_cv(
    tuner=SKBayesTuner(SKLogisticRegression()),
    search=space,
    name="Logistic Regression",
    duplicate=False,
    hard_mode=True,
)
```

### Duplication

```python
outer_cv(
    tuner=SKBayesTuner(DuplicateSKLogisticRegression()),
    search=space,
    name="Logistic Regression",
    duplicate=True,
    hard_mode=False,
)
```

### Duplication, Hard

```python
outer_cv(
    tuner=SKBayesTuner(DuplicateSKLogisticRegression()),
    search=space,
    name="Logistic Regression",
    duplicate=True,
    hard_mode=True,
)
```
