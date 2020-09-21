# accumulator-batching

`tf.data` batching with conditions checked per example. For example, graph neural networks might batch up to a maximum number of nodes and/or edges rather than a fixed batch size.

## Example usage

```python
import numpy as np
import tensorflow as tf

import accumulator_batching as acc

ragged_data = [np.arange(n, dtype=np.float32) for n in (2, 3, 4, 5, 6, 2, 1)]
rect_data = np.arange(len(ragged_data), dtype=np.float32)

dataset = tf.data.Dataset.from_generator(
    lambda: zip(ragged_data, rect_data), (tf.float32, tf.float32), ((None,), ())
)
ac = (
    acc.RaggedAccumulator(tf.TensorSpec((None,), dtype=tf.float32), max_flat_size=7),
    acc.TensorAccumulator(tf.TensorSpec((), dtype=tf.float32)),
)

dataset = acc.accumulated_batch(dataset, ac)

expected_flat_values = [
    ([0, 1, 0, 1, 2], [0, 1]),
    ([0, 1, 2, 3], [2]),
    ([0, 1, 2, 3, 4], [3]),
    ([0, 1, 2, 3, 4, 5], [4]),
    ([0, 1, 0], [5, 6]),
]

for actual, expected in zip(dataset, expected_flat_values):
    actual = (actual[0].flat_values.numpy(), actual[1].numpy())
    np.testing.assert_equal(actual, expected)
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
