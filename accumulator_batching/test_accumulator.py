import numpy as np
import tensorflow as tf
from tensorflow import test

import accumulator_batching as acc


class AccumulatorTest(test.TestCase):
    def test_tensor_batching(self):
        dataset = tf.data.Dataset.range(10, output_type=tf.float32)
        print(dataset.element_spec)
        ac = acc.TensorAccumulator(tf.TensorSpec((), dtype=tf.float32), max_size=3)
        dataset = acc.accumulated_batch(dataset, ac)

        expecteds = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        for actual, expected in zip(dataset, expecteds):
            np.testing.assert_equal(actual.numpy(), expected)

    def test_ragged_batching(self):
        data = [np.arange(n, dtype=np.float32) for n in (2, 3, 4, 5, 6, 2, 1)]
        ac = acc.RaggedAccumulator(tf.TensorSpec((None,), dtype=tf.float32), 7)

        dataset = tf.data.Dataset.from_generator(lambda: data, tf.float32, (None,))
        dataset = acc.accumulated_batch(dataset, ac)

        expecteds = [
            [0, 1, 0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 0],
        ]

        for actual, expected in zip(dataset, expecteds):
            np.testing.assert_equal(actual.flat_values.numpy(), expected)

    def test_tuple_batching(self):
        ragged_data = [np.arange(n, dtype=np.float32) for n in (2, 3, 4, 5, 6, 2, 1)]
        rect_data = np.arange(len(ragged_data), dtype=np.float32)

        dataset = tf.data.Dataset.from_generator(
            lambda: zip(ragged_data, rect_data), (tf.float32, tf.float32), ((None,), ())
        )
        ac = (
            acc.RaggedAccumulator(tf.TensorSpec((None,), dtype=tf.float32), 7),
            acc.TensorAccumulator(tf.TensorSpec((), dtype=tf.float32)),
        )

        dataset = acc.accumulated_batch(dataset, ac)

        expecteds = [
            ([0, 1, 0, 1, 2], [0, 1]),
            ([0, 1, 2, 3], [2]),
            ([0, 1, 2, 3, 4], [3]),
            ([0, 1, 2, 3, 4, 5], [4]),
            ([0, 1, 0], [5, 6]),
        ]

        for actual, expected in zip(dataset, expecteds):
            actual = (actual[0].flat_values.numpy(), actual[1].numpy())
            np.testing.assert_equal(actual, expected)

    def test_map_batching(self):
        ragged_data = [np.arange(n, dtype=np.float32) for n in (2, 3, 4, 5, 6, 2, 1)]
        rect_data = np.arange(len(ragged_data), dtype=np.float32)

        dataset = tf.data.Dataset.from_generator(
            lambda: (dict(a=a, b=b) for a, b in zip(ragged_data, rect_data)),
            dict(a=tf.float32, b=tf.float32),
            dict(a=(None,), b=()),
        )
        ac = dict(
            a=acc.RaggedAccumulator(tf.TensorSpec((None,), dtype=tf.float32), 7),
            b=acc.TensorAccumulator(tf.TensorSpec((), dtype=tf.float32)),
        )

        dataset = acc.accumulated_batch(dataset, ac)

        expecteds = [
            ([0, 1, 0, 1, 2], [0, 1]),
            ([0, 1, 2, 3], [2]),
            ([0, 1, 2, 3, 4], [3]),
            ([0, 1, 2, 3, 4, 5], [4]),
            ([0, 1, 0], [5, 6]),
        ]

        for actual, expected in zip(dataset, expecteds):
            actual = (actual["a"].flat_values.numpy(), actual["b"].numpy())
            np.testing.assert_equal(actual, expected)

    def test_repeated(self):
        data = [np.arange(n, dtype=np.float32) for n in (2, 3, 4, 5, 6, 2, 1)]
        ac = acc.RaggedAccumulator(tf.TensorSpec((None,), dtype=tf.float32), 7)

        dataset = tf.data.Dataset.from_generator(lambda: data, tf.float32, (None,))
        dataset = dataset.repeat()
        dataset = acc.accumulated_batch(dataset, ac)

        expecteds = [
            [0, 1, 0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 0, 0, 1],
        ]

        for actual, expected in zip(dataset, expecteds):
            np.testing.assert_equal(actual.flat_values.numpy(), expected)


if __name__ == "__main__":
    test.main()
