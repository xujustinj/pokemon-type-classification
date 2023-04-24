## `model/__init__.py`

```{py}
from .model import Model, Config
from .sk_model import (
    SKModel,
    SKClassifier,
    SKLogisticRegression,
    SKSupportVectorMachine,
    SKRandomForest,
)
from .duplication import (
    DuplicateSKLogisticRegression,
    DuplicateSKRandomForest,
    DuplicateSKSupportVectorMachine,
)
```
