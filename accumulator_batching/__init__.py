import abc
import itertools
from typing import (
    Any,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import tensorflow as tf

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("T")


class Accumulator(Generic[S, T, U]):
    @abc.abstractmethod
    def initial_state(self) -> S:
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def append(self, state: S, element: T) -> S:
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def finalize(self, state: S) -> U:
        raise NotImplementedError("Abstract method")

    def valid_conditions(self, state: S) -> Tuple[tf.Tensor, ...]:
        return ()


class TensorAccumulator(Accumulator[tf.Tensor, tf.Tensor, tf.Tensor]):
    def __init__(self, spec: tf.TensorSpec, max_size: Optional[int] = None):
        self._spec = spec
        self._max_size = max_size

    @property
    def spec(self) -> tf.TensorSpec:
        return self._spec

    @property
    def max_size(self) -> Optional[int]:
        return self._max_size

    def initial_state(self) -> tf.Tensor:
        return tf.zeros(shape=(0, *self._spec.shape), dtype=self._spec.dtype)

    def append(self, state: tf.Tensor, element: tf.Tensor) -> tf.Tensor:
        return tf.concat((state, tf.expand_dims(element, axis=0)), axis=0)

    def finalize(self, state: tf.Tensor) -> tf.Tensor:
        return state

    def valid_conditions(self, state: S) -> Tuple[tf.Tensor, ...]:
        if self._max_size is None:
            return ()
        return (tf.shape(state)[0] <= self._max_size,)


class RaggedComponents(NamedTuple):
    flat_values: tf.Tensor
    row_splits: tf.Tensor


class RaggedAccumulator(Accumulator[RaggedComponents, tf.Tensor, tf.RaggedTensor]):
    def __init__(
        self,
        flat_spec: tf.TensorSpec,
        max_flat_size: Optional[int] = None,
        row_splits_dtype=tf.int64,
    ):
        self._flat_spec = flat_spec
        self._row_splits_dtype = row_splits_dtype
        self._max_flat_size = max_flat_size

    @property
    def flat_spec(self) -> tf.TensorSpec:
        return self._flat_spec

    @property
    def row_splits_dtype(self) -> tf.DType:
        return self._row_splits_dtype

    @property
    def max_flat_size(self) -> Optional[int]:
        return self._max_flat_size

    def initial_state(self) -> RaggedComponents:
        return RaggedComponents(
            tf.zeros(
                [0 if s is None else s for s in self._flat_spec.shape],
                dtype=self._flat_spec.dtype,
            ),
            tf.zeros((1,), dtype=self._row_splits_dtype),
        )

    def append(self, state: RaggedComponents, element: tf.Tensor) -> RaggedComponents:
        flat_values, row_splits = state
        flat_values = tf.concat((flat_values, element), axis=0)
        row_splits = tf.concat(
            (
                row_splits,
                tf.expand_dims(
                    row_splits[-1]
                    + tf.shape(element, out_type=self._row_splits_dtype)[0],
                    axis=0,
                ),
            ),
            axis=0,
        )
        return RaggedComponents(flat_values, row_splits)

    def valid_conditions(self, state: RaggedComponents) -> Tuple[tf.Tensor, ...]:
        if self._max_flat_size is None:
            return ()
        return (state.row_splits[-1] <= self._max_flat_size,)

    def finalize(self, state: RaggedComponents) -> tf.Tensor:
        return tf.RaggedTensor.from_row_splits(*state)


class TupleAccumulator(Accumulator[Tuple, Tuple[tf.Tensor, ...], Tuple]):
    def __init__(self, *accumulators: Accumulator):
        self._accumulators = accumulators

    @property
    def accumulators(self) -> Tuple[Accumulator, ...]:
        return self._accumulators

    def initial_state(self) -> Tuple:
        return tuple(acc.initial_state() for acc in self._accumulators)

    def append(self, state: Tuple, element: Tuple[tf.Tensor, ...]) -> Tuple:
        assert len(state) == len(element) == len(self._accumulators)
        return tuple(
            acc.append(s, el)
            for (acc, s, el) in zip(self._accumulators, state, element)
        )

    def valid_conditions(self, state: Tuple) -> Tuple[tf.Tensor, ...]:
        assert len(state) == len(self._accumulators)
        return tuple(
            itertools.chain(
                *(acc.valid_conditions(s) for acc, s in zip(self._accumulators, state))
            )
        )

    def finalize(self, state: Tuple) -> Tuple:
        assert len(state) == len(self._accumulators)
        return tuple(acc.finalize(s) for acc, s in zip(self._accumulators, state))


class MapAccumulator(
    Accumulator[Mapping[str, Any], Mapping[str, tf.Tensor], Mapping[str, Any]]
):
    def __init__(self, **kwargs: Accumulator):
        self._accumulators = kwargs

    @property
    def accumulators(self) -> Mapping[str, Accumulator]:
        return self._accumulators

    def initial_state(self) -> Mapping[str, Any]:
        return {k: v.initial_state() for k, v in self._accumulators.items()}

    def append(
        self, state: Mapping[str, Any], element: Mapping[str, tf.Tensor]
    ) -> Mapping[str, Any]:
        assert len(element) == len(self._accumulators)
        return {
            k: v.append(state[k], element[k]) for k, v in self._accumulators.items()
        }

    def valid_conditions(self, state: Mapping[str, Any]) -> Tuple[tf.Tensor, ...]:
        return tuple(
            itertools.chain(
                *(v.valid_conditions(state[k]) for k, v in self._accumulators.items())
            )
        )

    def finalize(self, state: Mapping) -> Mapping:
        return {k: v.finalize(state[k]) for k, v in self._accumulators.items()}


def accumulator_structure(acc: Union[Accumulator, Mapping, Iterable]):
    if isinstance(acc, Accumulator):
        return acc
    elif isinstance(acc, Mapping):
        return MapAccumulator(**{k: accumulator_structure(v) for k, v in acc.items()})
    elif isinstance(acc, Iterable):
        return TupleAccumulator(*(accumulator_structure(v) for v in acc))
    else:
        raise TypeError(
            f"Invalid acc: must be Accumulator, Mapping or Iterable, got {type(acc)}"
        )


def accumulated_batch(
    dataset: tf.data.Dataset,
    accumulator: Union[Accumulator, Mapping, Iterable],
    **map_kwargs,
):
    accumulator = accumulator_structure(accumulator)

    def initial_map_fn(*args):
        if len(args) == 1:
            (args,) = args
        return args, False

    @tf.function
    def scan_fn(state, el_and_final):
        el, final = el_and_final
        new_state = accumulator.append(state, el)
        valid = tf.reduce_all(accumulator.valid_conditions(new_state))
        invalid = tf.logical_not(valid)
        if invalid:
            new_state = accumulator.append(accumulator.initial_state(), el)
        return new_state, (state, tf.logical_or(invalid, final))

    def filter_fn(state, invalid):
        del state
        return invalid

    def map_fn(state, invalid):
        del invalid
        return accumulator.finalize(state)

    cardinality = tf.data.experimental.cardinality(dataset)

    dataset = dataset.map(initial_map_fn)
    if cardinality != tf.data.experimental.INFINITE_CARDINALITY:
        # append (empty, True) element to ensure final elements are generated
        state_spec = dataset.element_spec[0]
        empty_el = tf.nest.map_structure(
            lambda spec: tf.zeros(
                [1, *(0 if s is None else s for s in spec.shape)], dtype=spec.dtype
            ),
            state_spec,
        )
        true_el = tf.ones((1,), dtype=tf.bool)
        dataset = dataset.concatenate(
            tf.data.Dataset.from_tensor_slices((empty_el, true_el))
        )

    dataset = dataset.apply(
        tf.data.experimental.scan(accumulator.initial_state(), scan_fn)
    )

    dataset = dataset.filter(filter_fn)
    dataset = dataset.map(map_fn, **map_kwargs)
    return dataset
