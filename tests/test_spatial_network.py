"""The shared spatial policy, value and Q network for the Stoix baselines."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import hydra.utils
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import jaxarc
from jaxarc import JaxArcConfig
from jaxarc.envs.grid_operations import SELECTION_INDEPENDENT_OPERATIONS
from jaxarc.registration import available_task_ids
from jaxarc.types import ARCOperationType
from jaxarc.stoix_adapter import make_jaxarc_env
from jaxarc_baselines.spatial_network import (
    PADDING_CATEGORY,
    ColourEmbedding,
    SpatialPolicyHead,
    SpatialQHead,
    SpatialTorso,
    SpatialValueHead,
    colour_categories,
)
from stoix.networks.base import FeedForwardCritic

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "experiments" / "configs"
NETWORKS = ROOT / "configs" / "network"
ACTION_CONFIGS = Path(jaxarc.__file__).parent / "conf" / "action"
PROFILES = {
    name: yaml.safe_load((ACTION_CONFIGS / f"{name}.yaml").read_text())[
        "allowed_operations"
    ]
    for name in ("raw", "full")
}
CHANNELS = 12  # working grid, input grid and five context pairs


def observation(size, batch=3, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.randint(key, (batch, size, size, CHANNELS), -1, 10).astype(
        jnp.float32
    )


def action_count(profile, size):
    ops = PROFILES[profile]
    independent = [op for op in ops if op in SELECTION_INDEPENDENT_OPERATIONS]
    return (len(ops) - len(independent)) * size * size + len(independent)


def network_config(name, profile="raw"):
    """A network config with the interpolations it uses resolved."""
    config = OmegaConf.create(
        {
            "network": yaml.safe_load((NETWORKS / f"{name}.yaml").read_text()),
            "action": {"allowed_operations": PROFILES[profile]},
        }
    )
    return config.network


# --- The action layout ------------------------------------------------------


def flat_action_decoder(dataset, scenario_dataset, profile):
    """For every flat action, the operation and cell the env's wrapper runs.

    The working grid is set to the whole canvas so the selection is the clicked
    cell, not the cell clipped to the active grid.
    """
    task_id = available_task_ids(scenario_dataset, config=JaxArcConfig())[0]
    with initialize_config_dir(str(CONFIGS), version_base="1.2"):
        config = compose(
            "default_ppo_jaxarc",
            overrides=[
                f"dataset={dataset}",
                f"action={profile}",
                f"env.scenario.name={scenario_dataset}-{task_id}",
            ],
        )
    OmegaConf.set_struct(config, False)
    env, _ = make_jaxarc_env(config)
    state, timestep = env.reset(jax.random.PRNGKey(0))
    state = eqx.tree_at(
        lambda s: s.working_grid_mask, state, jnp.ones_like(state.working_grid_mask)
    )

    def decode(action):
        chosen = env.step(state, action)[1].extras["canonical_action"]
        return chosen["operation"], jnp.argmax(chosen["selection"].reshape(-1))

    actions = jnp.arange(int(env.action_space().num_values))
    operation, cell = jax.jit(lambda a: jax.lax.map(decode, a, batch_size=2048))(
        actions
    )
    return np.asarray(operation), np.asarray(cell), timestep.observation.shape[:2]


@pytest.fixture(
    scope="module",
    params=[
        ("mini_arc", "Mini", "raw"),
        ("mini_arc", "Mini", "full"),
        ("arc_agi_1", "AGI1", "raw"),
        ("arc_agi_1", "AGI1", "full"),
    ],
    ids=["5x5-raw", "5x5-full", "30x30-raw", "30x30-full"],
)
def decoded_actions(request):
    dataset, scenario_dataset, profile = request.param
    if not available_task_ids(scenario_dataset, config=JaxArcConfig()):
        pytest.skip(f"No {scenario_dataset} tasks available")
    operation, cell, (height, width) = flat_action_decoder(
        dataset, scenario_dataset, profile
    )
    return profile, operation, cell, height, width


def labelled_scores(head, allowed, height, width):
    """Head output where each logit names the (operation, cell) it came from.

    Cell features are (row, col, 1). The 1x1 conv gives slot s at cell (r, c)
    the score s*H*W + r*W + c; the pooled MLP gives operation op the score -(op + 1).
    """
    rows, cols = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    features = jnp.stack([rows, cols, jnp.ones_like(rows)], axis=-1)[None].astype(
        jnp.float32
    )
    params = head.init(jax.random.PRNGKey(0), features)["params"]

    dependent = [op for op in allowed if op not in SELECTION_INDEPENDENT_OPERATIONS]
    independent = [op for op in allowed if op in SELECTION_INDEPENDENT_OPERATIONS]
    slots = jnp.arange(len(dependent), dtype=jnp.float32)
    scores = params["scores"]
    scores["cell"]["kernel"] = jnp.stack(
        [jnp.full_like(slots, width), jnp.ones_like(slots), slots * height * width]
    )[None, None]
    scores["cell"]["bias"] = jnp.zeros_like(slots)
    scores["global_out"]["kernel"] = jnp.zeros_like(scores["global_out"]["kernel"])
    scores["global_out"]["bias"] = -(jnp.asarray(independent, dtype=jnp.float32) + 1)

    output = head.apply({"params": params}, features)
    if isinstance(head, SpatialQHead):
        return np.asarray(output.preferences[0])
    return np.asarray(output.logits[0])


@pytest.mark.parametrize("head_type", [SpatialPolicyHead, SpatialQHead])
def test_every_logit_lines_up_with_the_action_the_env_runs(decoded_actions, head_type):
    profile, operation, cell, height, width = decoded_actions
    allowed = PROFILES[profile]
    head = head_type(action_dim=len(operation), allowed_operations=allowed)

    scores = labelled_scores(head, allowed, height, width)

    dependent = [op for op in allowed if op not in SELECTION_INDEPENDENT_OPERATIONS]
    expected = np.array(
        [
            (
                -(op + 1)
                if op in SELECTION_INDEPENDENT_OPERATIONS
                else dependent.index(op) * height * width + c
            )
            for op, c in zip(operation.tolist(), cell.tolist())
        ],
        dtype=np.float32,
    )
    assert scores.shape == (action_count(profile, height),)
    np.testing.assert_array_equal(scores, expected)


def test_raw_ops_give_one_submit_action():
    assert action_count("raw", 5) == 276
    assert action_count("raw", 30) == 9901


def test_a_head_built_for_another_action_count_is_rejected():
    head = SpatialPolicyHead(action_dim=300, allowed_operations=PROFILES["raw"])
    with pytest.raises(ValueError, match="276"):
        head.init(jax.random.PRNGKey(0), jnp.zeros((1, 5, 5, 8)))


def test_no_allowed_operations_means_every_operation():
    head = SpatialPolicyHead(action_dim=33 * 25 + 2, allowed_operations=None)
    params = head.init(jax.random.PRNGKey(0), jnp.zeros((1, 5, 5, 8)))

    assert head.apply(params, jnp.zeros((1, 5, 5, 8))).logits.shape == (1, 33 * 25 + 2)


# --- Shapes and grid sizes ----------------------------------------------------


def actor_critic(name, size, profile="raw"):
    """The actor and critic (or shared) modules a network config builds for one grid size."""
    from stoix.networks.base import (
        FeedForwardActor,
        FeedForwardActorCritic,
        FeedForwardCritic,
    )

    config = network_config(name, profile)
    action_dim = action_count(profile, size)
    head = hydra.utils.instantiate(
        config.actor_network.action_head, action_dim=action_dim
    )
    if "shared_network" in config:
        torso = hydra.utils.instantiate(config.shared_network.pre_torso)
        critic_head = hydra.utils.instantiate(config.critic_network.critic_head)
        return FeedForwardActorCritic(
            torso=torso, action_head=head, critic_head=critic_head
        )
    actor = FeedForwardActor(
        torso=hydra.utils.instantiate(config.actor_network.pre_torso), action_head=head
    )
    critic = FeedForwardCritic(
        torso=hydra.utils.instantiate(config.critic_network.pre_torso),
        critic_head=hydra.utils.instantiate(config.critic_network.critic_head),
    )
    return actor, critic


def test_separate_network_works_at_5x5_and_30x30_with_the_same_params():
    small_actor, small_critic = actor_critic("arc_spatial", 5)
    big_actor, big_critic = actor_critic("arc_spatial", 30)
    actor_params = small_actor.init(jax.random.PRNGKey(0), observation(5))
    critic_params = small_critic.init(jax.random.PRNGKey(1), observation(5))

    for size, actor, critic in (
        (5, small_actor, small_critic),
        (30, big_actor, big_critic),
    ):
        obs = observation(size)
        policy = actor.apply(actor_params, obs)
        value = critic.apply(critic_params, obs)

        assert policy.logits.shape == (3, action_count("raw", size))
        assert policy.logits.dtype == jnp.float32
        assert value.shape == (3,) and value.dtype == jnp.float32
        assert policy.sample(seed=jax.random.PRNGKey(2)).shape == (3,)


def test_shared_network_works_at_5x5_and_30x30_with_the_same_params():
    small, big = actor_critic("arc_spatial_shared", 5), actor_critic(
        "arc_spatial_shared", 30
    )
    params = small.init(jax.random.PRNGKey(0), observation(5))

    for size, network in ((5, small), (30, big)):
        policy, value = network.apply(params, observation(size))
        assert policy.logits.shape == (3, action_count("raw", size))
        assert value.shape == (3,)


def test_pqn_network_gives_one_q_value_per_action_at_both_sizes():
    from stoix.networks.base import FeedForwardActor

    config = network_config("arc_spatial_pqn", "full")
    torso = hydra.utils.instantiate(config.actor_network.pre_torso)

    def q_network(size):
        head = hydra.utils.instantiate(
            config.actor_network.action_head,
            action_dim=action_count("full", size),
            epsilon=0.1,
        )
        return FeedForwardActor(torso=torso, action_head=head)

    params = q_network(5).init(jax.random.PRNGKey(0), observation(5))
    for size in (5, 30):
        q = q_network(size).apply(params, observation(size)).preferences
        assert q.shape == (3, action_count("full", size)) and q.dtype == jnp.float32


def test_torso_keeps_the_grid_and_its_params_do_not_depend_on_it():
    torso = SpatialTorso(channels=16, num_blocks=2)
    params = torso.init(jax.random.PRNGKey(0), observation(5))
    shapes = lambda p: jax.tree.map(jnp.shape, p)

    assert torso.apply(params, observation(7)).shape == (3, 7, 7, 16)
    assert shapes(params) == shapes(torso.init(jax.random.PRNGKey(0), observation(30)))


def test_value_head_pools_to_one_value_per_sample():
    head = SpatialValueHead()
    features = jnp.ones((4, 6, 6, 16))
    params = head.init(jax.random.PRNGKey(0), features)

    assert head.apply(params, features).shape == (4,)


# --- Colour input -----------------------------------------------------------


def test_padding_is_its_own_colour_category():
    grid = jnp.array([[-1.0, 0.0], [5.0, 9.0]])

    np.testing.assert_array_equal(
        colour_categories(grid), [[PADDING_CATEGORY, 0], [5, 9]]
    )
    assert PADDING_CATEGORY == 10


def test_embedding_has_eleven_categories_per_channel():
    torso = SpatialTorso(channels=16, num_blocks=1, colour_input="embed", embed_dim=4)
    params = torso.init(jax.random.PRNGKey(0), observation(5))["params"]

    assert params["colour_embedding"]["embedding"].shape == (11 * CHANNELS, 4)


def test_embedding_reads_row_11c_plus_colour_for_channel_c():
    embed = ColourEmbedding(embed_dim=4)
    obs = observation(5, batch=2)
    params = embed.init(jax.random.PRNGKey(0), obs)
    table = params["params"]["embedding"]
    rows = colour_categories(obs) + jnp.arange(CHANNELS) * 11

    expected = table[rows].reshape(2, 5, 5, CHANNELS * 4)
    np.testing.assert_allclose(embed.apply(params, obs), expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_embedding_gradient_has_no_scatter(dtype):
    # A gather's gradient is a scatter-add onto 132 rows, which is ~40x slower
    # than the rest of the torso in bfloat16 on an H100.
    torso = SpatialTorso(channels=8, num_blocks=1, dtype=dtype)
    obs = observation(5)
    params = torso.init(jax.random.PRNGKey(0), obs)
    loss = lambda p: torso.apply(p, obs).astype(jnp.float32).sum()

    assert "scatter" not in str(jax.make_jaxpr(jax.grad(loss))(params))


def test_padding_embeds_differently_from_every_colour():
    torso = SpatialTorso(channels=16, num_blocks=1, colour_input="embed")
    params = torso.init(jax.random.PRNGKey(0), observation(5))
    out = lambda value: torso.apply(params, jnp.full((1, 5, 5, CHANNELS), value))

    padding = out(-1.0)
    for colour in range(10):
        assert not jnp.allclose(out(float(colour)), padding)


def test_raw_input_reads_the_observation_values():
    torso = SpatialTorso(channels=16, num_blocks=1, colour_input="raw")
    params = torso.init(jax.random.PRNGKey(0), observation(5))["params"]

    assert "colour_embedding" not in params
    assert params["Conv_0"]["kernel"].shape == (3, 3, CHANNELS, 16)


def test_unknown_switch_values_are_rejected():
    with pytest.raises(ValueError, match="colour_input"):
        SpatialTorso(colour_input="onehot").init(jax.random.PRNGKey(0), observation(5))
    with pytest.raises(ValueError, match="dtype"):
        SpatialTorso(dtype="float16").init(jax.random.PRNGKey(0), observation(5))


# --- Precision --------------------------------------------------------------


@pytest.mark.parametrize("colour_input", ["embed", "raw"])
def test_bfloat16_keeps_float32_params_and_outputs_close_to_float32(colour_input):
    from stoix.networks.base import FeedForwardActorCritic

    def network(dtype):
        return FeedForwardActorCritic(
            torso=SpatialTorso(colour_input=colour_input, dtype=dtype),
            action_head=SpatialPolicyHead(
                action_dim=276, allowed_operations=PROFILES["raw"]
            ),
            critic_head=SpatialValueHead(),
        )

    obs = observation(5, batch=8)
    f32, bf16 = network("float32"), network("bfloat16")
    params = f32.init(jax.random.PRNGKey(0), obs)

    assert all(
        p.dtype == jnp.float32
        for p in jax.tree.leaves(bf16.init(jax.random.PRNGKey(0), obs))
    )
    policy32, value32 = f32.apply(params, obs)
    policy16, value16 = bf16.apply(params, obs)
    assert policy16.logits.dtype == jnp.float32 and value16.dtype == jnp.float32
    np.testing.assert_allclose(
        jax.nn.log_softmax(policy16.logits),
        jax.nn.log_softmax(policy32.logits),
        atol=2e-2,
    )
    # At init a raw-input torso can have bf16 value errors of several percent of
    # the value scale; the embedding torso stays near 1%.
    tolerance = {"embed": 5e-2, "raw": 1e-1}[colour_input]
    np.testing.assert_allclose(
        value16, value32, atol=tolerance * max(1.0, float(jnp.abs(value32).max()))
    )


def test_bfloat16_torso_computes_in_bfloat16():
    torso = SpatialTorso(channels=16, num_blocks=1, dtype="bfloat16")
    params = torso.init(jax.random.PRNGKey(0), observation(5))

    assert torso.apply(params, observation(5)).dtype == jnp.bfloat16


# --- Configs ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["arc_spatial", "arc_spatial_shared", "arc_spatial_pqn"]
)
def test_network_configs_default_to_a_small_bfloat16_embedding_torso(name):
    # 64 x 2 in bfloat16 was the fastest size that learned on MiniARC (results in
    # .scratch/tmlr-jaxarc-revision/step4/results.md).
    torso = network_config(name).torso

    assert torso.colour_input == "embed"
    assert torso.dtype == "bfloat16"
    assert (torso.channels, torso.num_blocks) == (64, 2)


def test_torso_class_defaults_match_the_configs():
    torso = SpatialTorso()

    assert (torso.channels, torso.num_blocks, torso.dtype) == (64, 2, "bfloat16")


def test_one_override_switches_both_separate_torsos():
    config = network_config("arc_spatial")
    config.torso.dtype = "float32"

    assert config.actor_network.pre_torso.dtype == "float32"
    assert config.critic_network.pre_torso.dtype == "float32"


def test_head_reads_the_operations_from_the_action_config():
    config = network_config("arc_spatial", "full")

    assert list(config.actor_network.action_head.allowed_operations) == PROFILES["full"]


@pytest.mark.parametrize("head", ["policy", "q", "value"])
def test_heads_compute_in_float32_on_bfloat16_features(head):
    features = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 5, 8)).astype(
        jnp.bfloat16
    )
    module = {
        "policy": SpatialPolicyHead(action_dim=276, allowed_operations=PROFILES["raw"]),
        "q": SpatialQHead(action_dim=276, allowed_operations=PROFILES["raw"]),
        "value": SpatialValueHead(),
    }[head]
    params = module.init(jax.random.PRNGKey(1), features)
    scores = lambda out: out if head == "value" else (
        out.logits if head == "policy" else out.preferences
    )

    from_bf16 = scores(module.apply(params, features))
    from_f32 = scores(module.apply(params, features.astype(jnp.float32)))

    np.testing.assert_array_equal(from_bf16, from_f32)


def test_value_head_starts_near_zero_for_every_seed():
    # Under pbrs a wrong submit pays minus the current similarity. A critic that
    # starts optimistic makes every early submit look bad, and PPO stopped
    # submitting on such seeds (results.md, "Seed-2 collapse").
    obs = observation(5, batch=16)
    critic = FeedForwardCritic(
        torso=SpatialTorso(channels=16, num_blocks=1), critic_head=SpatialValueHead()
    )
    values = [
        critic.apply(critic.init(jax.random.PRNGKey(seed), obs), obs)
        for seed in range(8)
    ]

    assert float(jnp.abs(jnp.stack(values)).max()) < 0.05


# --- Init options for the never-submit collapse (strategy-v4 §10d) ------------


def submit_index(profile, size):
    allowed = PROFILES[profile]
    dependent = [op for op in allowed if op not in SELECTION_INDEPENDENT_OPERATIONS]
    independent = [op for op in allowed if op in SELECTION_INDEPENDENT_OPERATIONS]
    return len(dependent) * size * size + independent.index(ARCOperationType.SUBMIT)


def test_value_bias_init_sets_the_starting_value_for_every_seed():
    obs = observation(5, batch=16)
    critic = FeedForwardCritic(
        torso=SpatialTorso(channels=16, num_blocks=1),
        critic_head=SpatialValueHead(bias_init=-1.0),
    )
    values = jnp.stack(
        [critic.apply(critic.init(jax.random.PRNGKey(seed), obs), obs) for seed in range(8)]
    )

    assert float(jnp.abs(values + 1.0).max()) < 0.05


@pytest.mark.parametrize("profile", ["raw", "full"])
def test_submit_bias_shifts_only_the_submit_logit_at_init(profile):
    features = jax.random.normal(jax.random.PRNGKey(3), (2, 5, 5, 16))
    action_dim = action_count(profile, 5)
    plain = SpatialPolicyHead(action_dim=action_dim, allowed_operations=PROFILES[profile])
    biased = SpatialPolicyHead(
        action_dim=action_dim, allowed_operations=PROFILES[profile], submit_bias=1.0
    )
    key = jax.random.PRNGKey(0)
    diff = (
        biased.apply(biased.init(key, features), features).logits
        - plain.apply(plain.init(key, features), features).logits
    )
    expected = np.zeros(action_dim, dtype=np.float32)
    expected[submit_index(profile, 5)] = 1.0

    np.testing.assert_allclose(diff, np.broadcast_to(expected, diff.shape), atol=1e-6)


def test_submit_bias_without_submit_is_rejected():
    allowed = [op for op in PROFILES["raw"] if op != ARCOperationType.SUBMIT]
    independent = [op for op in allowed if op in SELECTION_INDEPENDENT_OPERATIONS]
    head = SpatialPolicyHead(
        action_dim=(len(allowed) - len(independent)) * 25 + len(independent),
        allowed_operations=allowed,
        submit_bias=1.0,
    )
    with pytest.raises(ValueError, match="Submit"):
        head.init(jax.random.PRNGKey(0), jnp.zeros((1, 5, 5, 8)))


@pytest.mark.parametrize(
    ("name", "submit_bias", "bias_init"),
    [
        ("arc_spatial", 0.0, 0.0),
        # Shared torso ships with both on: it stopped the never-submit collapse.
        ("arc_spatial_shared", 1.0, -0.5),
    ],
)
def test_init_options_have_their_defaults_and_reach_the_heads(name, submit_bias, bias_init):
    config = network_config(name)
    assert config.actor_network.action_head.submit_bias == submit_bias
    assert config.critic_network.critic_head.bias_init == bias_init

    config.actor_network.action_head.submit_bias = 1.0
    config.critic_network.critic_head.bias_init = -1.0
    head = hydra.utils.instantiate(config.actor_network.action_head, action_dim=276)
    critic_head = hydra.utils.instantiate(config.critic_network.critic_head)

    assert head.submit_bias == 1.0
    assert critic_head.bias_init == -1.0
