import difflib
import functools
import operator
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import chex
import jax
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from hydra import compose, initialize
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from pytest_regressions.file_regression import FileRegressionFixture

from purejaxql.pqn_gymnax_flat import Config


@pytest.fixture(autouse=True)
def use_harsher_jax_jit_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Temporarily make this particular warning into an error to help future-proof our jax code.
    import jax._src.deprecations

    monkeypatch.setattr(
        jax._src.deprecations._registered_deprecations["tracer-hash"],
        "accelerated",
        True,
    )


@pytest.fixture
def command_line_overrides(request: pytest.FixtureRequest) -> list[str]:
    return list(getattr(request, "param", []))


# note: default number of timesteps in the config is 5e5, making it shorter to keep tests fast.
def use_fewer_timesteps(
    total_timesteps: int = 200,
    num_steps: int = 10,
    num_envs: int = 10,
    num_updates: int = 2,
):
    assert num_updates >= 1
    assert num_steps >= 1
    assert num_envs >= 1
    assert total_timesteps >= 1
    assert num_updates == total_timesteps // num_steps // num_envs

    return pytest.mark.parametrize(
        command_line_overrides.__name__,
        [
            [
                # (num updates := total timesteps // num steps // num envs)
                f"alg.TOTAL_TIMESTEPS={total_timesteps}",
                f"alg.TOTAL_TIMESTEPS_DECAY={total_timesteps}",
                f"alg.NUM_STEPS={num_steps}",
                f"alg.NUM_ENVS={num_envs}",
            ]
        ],
        indirect=True,
    )


@pytest.fixture
def config_name(request: pytest.FixtureRequest):
    """The config name to use during tests, ex 'pqn_cartpole'.

    Should be supplied using indirect parametrization like this:
    ```python
    @pytest.mark.parametrize("config_name", ["pqn_cartpole"], indirect=True)
    def some_test(config: dict):
        ...
    """
    return request.param


def use_alg_configs(alg_config_name: str | list[str]):
    return pytest.mark.parametrize(
        config_name.__name__,
        [alg_config_name] if isinstance(alg_config_name, str) else alg_config_name,
        indirect=True,
    )


@pytest.fixture
def config(
    config_name: str, command_line_overrides: list[str], request: pytest.FixtureRequest
):
    with initialize(version_base="1.2", config_path="../purejaxql/config"):
        overrides = [f"+alg={config_name}", *command_line_overrides]
        print(
            "Running test with the same config as if this was passed on the command-line:"
        )
        # Figure out which script is being tested with `request.module.__file__`
        module_under_test = (
            Path(request.module.__file__)
            .stem.removeprefix("test_")
            .removesuffix("test")
        )
        print(f"python purejaxql/{module_under_test} {' '.join(overrides)}")
        _config = compose(
            config_name="config",
            overrides=overrides,
        )
        config: Config = {**_config, **_config["alg"]}  # type: ignore

        from purejaxql.pqn_gymnax_flat import (
            _get_num_updates,
            _get_num_updates_decay,
        )

        # Make sure that we'll be able to do at least one update with this config.
        assert _get_num_updates(config) >= 1
        assert _get_num_updates_decay(config) >= 1

        yield config


@pytest.fixture(params=[42, 123])
def seed(request: pytest.FixtureRequest) -> int:
    return getattr(request, "param")


@pytest.fixture(params=[None, 5])
def num_seeds(request: pytest.FixtureRequest) -> int | None:
    return getattr(request, "param")


@pytest.fixture(params=[True, False], ids=["jit", "no_jit"])
def jit(request: pytest.FixtureRequest) -> bool:
    return getattr(request, "param")


def assert_results_not_empty(results: Mapping[str, Any]) -> None:
    assert results["metrics"]
    shapes = jax.tree.map(np.shape, results["metrics"])
    assert all(v and v != (0,) for v in jax.tree.leaves(shapes)), shapes


class AlgoTests:
    """Simple tests for a pqn module."""

    make_train: Callable[[Any], Callable[[chex.PRNGKey], Any]]

    def test_train_is_reproducible(
        self,
        ndarrays_regression: NDArraysRegressionFixture,
        config: Config,
        jit: bool,
        seed: int,
        num_seeds: int | None,
    ):
        """Test that the results of `train` are reproduble for the same seed given the same hardware config."""
        train_fn = type(self).make_train(config)
        if num_seeds is not None:
            train_fn = jax.vmap(train_fn)
        if jit:
            train_fn = jax.jit(train_fn)

        rng = jax.random.PRNGKey(seed)
        if num_seeds is not None:
            rng = jax.random.split(rng, num_seeds)

        outputs = train_fn(rng)
        print(jax.tree.map(np.shape, outputs))
        # TODO: there seems to be a tuple somewhere in the runner_state entry of the outputs.
        ndarrays_regression.check(
            flatten_dict(
                jax.tree.map(operator.methodcaller("__array__"), outputs["metrics"]),
                sep=".",
            )
        )


class ComparisonTests:
    """Tests for comparing the results of two different implementations of the same algorithm."""

    make_train: Callable[[Any], Callable[[chex.PRNGKey], Any]]
    original_make_train: Callable[[Any], Callable[[Any], Mapping]]

    def test_results_are_identical(
        self,
        config: Config,
        jit: bool,
        seed: int,
        num_seeds: int | None,
        file_regression: FileRegressionFixture,
    ):
        # Interesting that this test fails when `jit=False`!
        original_train_fn = type(self).original_make_train(config)
        new_train_fn = type(self).make_train(config)

        if num_seeds is not None:
            original_train_fn = jax.vmap(original_train_fn)
            new_train_fn = jax.vmap(new_train_fn)

        if jit:
            original_train_fn = jax.jit(original_train_fn)
            new_train_fn = jax.jit(new_train_fn)

        rng = jax.random.PRNGKey(seed)
        if num_seeds is not None:
            rng = jax.random.split(rng, num_seeds)

        original_results = original_train_fn(rng)
        flattened_results = new_train_fn(rng)
        original_results_structure = jax.tree.structure(original_results)
        flattened_results_structure = jax.tree.structure(flattened_results)
        # Showing the diff can be useful while debugging if the structure of the results is unexpectedly different.
        assert (
            original_results_structure.num_leaves
            == flattened_results_structure.num_leaves
        ), "".join(
            difflib.context_diff(
                str(original_results_structure).replace(", ", ",\n"),
                str(flattened_results_structure).replace(", ", ",\n"),
                fromfile="original structure",
                tofile="flattened structure",
            )
        )

        leaves_are_equal = jax.tree.map(
            # check that results are equal with a bit of wiggle room when `jit=False`.
            functools.partial(
                np.allclose, rtol=1e-07 if jit else 1e-05, atol=0 if jit else 1e-6
            ),
            jax.tree.leaves(original_results),
            jax.tree.leaves(flattened_results),
        )
        equals_pytree = jax.tree.unflatten(original_results_structure, leaves_are_equal)
        assert all(jax.tree.leaves(equals_pytree)), equals_pytree
