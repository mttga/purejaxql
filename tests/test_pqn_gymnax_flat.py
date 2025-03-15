import pytest

import purejaxql
import purejaxql.pqn_gymnax
import purejaxql.pqn_gymnax_flat

from .conftest import AlgoTests, ComparisonTests, use_alg_configs, use_fewer_timesteps


@use_alg_configs(["pqn_cartpole"])
@use_fewer_timesteps(
    num_updates=2,
    num_steps=64,
    num_envs=32,
    total_timesteps=32 * 64 * 2,
)
class TestFlattenedPqnGymnax(AlgoTests, ComparisonTests):
    original_make_train = purejaxql.pqn_gymnax.make_train
    make_train = purejaxql.pqn_gymnax_flat.make_train

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
        config: purejaxql.pqn_gymnax_flat.Config,
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
