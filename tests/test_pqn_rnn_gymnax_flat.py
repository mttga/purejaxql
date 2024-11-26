import purejaxql
import purejaxql.pqn_rnn_gymnax
import purejaxql.pqn_rnn_gymnax_flat

import pytest
from .conftest import (
    AlgoTests,
    ComparisonTests,
    use_alg_configs,
    use_fewer_timesteps,
)

# note: default number of timesteps in the config is 5e5, we make this shorter for tests to stay fast.


@use_alg_configs(["pqn_rnn_cartpole", "pqn_rnn_memory_chain"])
@use_fewer_timesteps(
    num_updates=2,
    num_steps=64,
    num_envs=32,
    total_timesteps=32 * 64 * 2,
)
class TestFlattenedPqnRNNGymnax(AlgoTests, ComparisonTests):
    original_make_train = purejaxql.pqn_rnn_gymnax.make_train
    make_train = purejaxql.pqn_rnn_gymnax_flat.make_train

    @pytest.mark.parametrize(
        "jit",
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="TODO: results aren't precisely the same when jit=False.",
                ),
            ),
        ],
        ids=["jit", "no_jit"],
    )
    def test_results_are_identical(
        self,
        config: purejaxql.pqn_rnn_gymnax_flat.Config,
        jit: bool,
        seed: int,
        num_seeds: int | None,
        file_regression,
    ):
        super().test_results_are_identical(
            config=config,
            jit=jit,
            seed=seed,
            num_seeds=num_seeds,
            file_regression=file_regression,
        )
