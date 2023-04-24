## Support Vector Machine

Support vector machines with $K > 2$ classes are broken into two "flavours":
- **one-versus-rest**: Fit a separate SVM for each class against the rest, and classify according to the highest score.
- **one-versus-one**: Fit a separate SVM for each pair of classes, and classify according to the class that wins the most.

We choose to focus on one-versus-rest, because it fits fewer models (thus fewer parameters).

### Setup

```python
from model import SKSupportVectorMachine, DuplicateSKSupportVectorMachine
from tune import (
    Categorical,
    Constant,
    Integer,
    outer_cv,
    Real,
    SKBayesTuner,
)
```

### Search Space

For tuning, `sklearn` provides us 4 kernels, each slightly differently parameterized.
Of these, we consider the linear, polynomial, and radial basis function kernels.
The sigmoid kernel is [never guaranteed to be positive semi-definite](https://search.r-project.org/CRAN/refmans/kernlab/html/dots.html) so we avoid using it. Moreover, the linear kernel is just a special case of the polynomial kernel (`degree = 1`, `gamma = 1`, `coef0 = 0`) so we can specify it together with the polynomial kernel.

```python
space = [
    dict(
        # constants
        probability=Constant(True),
        random_state=Constant(441),
        decision_function_shape=Constant("ovr"),

        kernel=Constant("poly"),
        # variables
        degree=Integer(low=2, high=5, prior="log-uniform"),
        C=Real(low=1e-4, high=1e4, prior="log-uniform"),
        gamma=Real(low=1e-3, high=1e3, prior="log-uniform"),
        coef0=Categorical([0.0, 1.0]),  # effect of coef0 scales with gamma
    ), 
    dict(
        # constants
        probability=Constant(True),
        random_state=Constant(441),
        decision_function_shape=Constant("ovr"),
        
        kernel=Constant("rbf"),
        # variables
        C=Real(low=1e-4, high=1e4, prior="log-uniform"),
        gamma=Real(low=1e-3, high=1e3, prior="log-uniform"),
    ),
]
```

### Base

```python
outer_cv(
    tuner=SKBayesTuner(SKSupportVectorMachine()),
    search=space,
    name="SVM",
    duplicate=False,
    hard_mode=False,
)
```

### Hard

```python
outer_cv(
    tuner=SKBayesTuner(SKSupportVectorMachine()),
    search=space,
    name="SVM",
    duplicate=False,
    hard_mode=True,
)
```

### Duplication

```python
outer_cv(
    tuner=SKBayesTuner(DuplicateSKSupportVectorMachine()),
    search=space,
    name="SVM",
    duplicate=True,
    hard_mode=False,
)
```

### Duplication, Hard

```python
outer_cv(
    tuner=SKBayesTuner(DuplicateSKSupportVectorMachine()),
    search=space,
    name="SVM",
    duplicate=True,
    hard_mode=True,
)
```
