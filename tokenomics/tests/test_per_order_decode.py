"""Choice decode and validity layer (add-m4-orderer-redesign task 4.1, design D4, ADR-0059).

Spec scenarios (learned-orderer): "Whole choice", "Ineligible producer chosen", "Conservation
under bursts", "Legacy checkpoint replay" (the goldens in test_legacy_invariance.py, unedited),
plus: the env config fingerprint changes with `ordering_decode` and is unchanged at the default,
`pettingzoo` `api_test` on both decodes, overflow orders keep the legacy split, and an invalid
choice reaches `contract_failures.csv` as a `job` row through the gate's job path.
Every episode here is horizon ≤ 12, one seed."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pettingzoo.test import api_test

import tokenomics.env.aec_env as aec_env
import tokenomics.eval.joint_stack as joint_stack
from tokenomics.domain.config import Configuration, load_config
from tokenomics.domain.order import OrderInfo
from tokenomics.domain.ordering_agent import OrderingAgentCommand, OrderingAgentView
from tokenomics.domain.producing_agent import ProducerInfo
from tokenomics.env import TokenomicsAECEnv
from tokenomics.env.allocator import (
    SHARE_SUM_TOL,
    InvalidChoice,
    _allocate_int,
    allocate_suborders,
    validate_choice,
)
from tokenomics.env.decode import decode_ordering
from tokenomics.env.spaces import (
    ACTION_LOGIT_BOUND,
    CHOICE_V1,
    LOGITS_V1,
    ChoiceShareSpace,
    EnvSpec,
    build_spec,
    config_fingerprint,
    decode_pins,
)
from tokenomics.eval.baselines import ORDERING_BASELINES, PRODUCING_BASELINES
from tokenomics.eval.contract import CHOICE, FAILURE_COLUMNS, job_failure
from tokenomics.eval.gate import side_effect_params_json
from tokenomics.eval.gate_cli import main, read_rows, scaled_entry
from tokenomics.eval.registry import SEED_ORDER, Registry, build_worlds_v1, load_registry, save_registry
from tokenomics.eval.roster import Roster, RosterMember, contract_for, save_roster
from tokenomics.eval.runner import run_episode
from tokenomics.eval.scale import build_scale_config

ROOT = Path(__file__).resolve().parents[1]
COMPOSITE = ROOT / "tests" / "fixtures" / "composite.json"
PHASE1 = ROOT / "tokenomics" / "config" / "phase1.json"
REGISTRY_V2 = ROOT / "tokenomics" / "eval" / "registry" / "worlds_v2.json"
CALM = "dev-sparse_wide-12-calm-01"
SHOCKED = "dev-sparse_wide-12-shocked-00"
HORIZON = 12


def _calm() -> Configuration:
    return build_worlds_v1(SEED_ORDER).entry(CALM).config()


def _shocked() -> Configuration:
    return load_registry(REGISTRY_V2).entry(SHOCKED).config()


def _ids(n: int) -> list[str]:
    return [f"p{i}" for i in range(n)]


# --- fingerprint and pins ---

# Legacy fingerprints computed on the base commit e52b362 (before `ordering_decode` existed):
# (config, orderer_obs_version, producer_obs_version) -> config_fingerprint.
BASE_FINGERPRINTS = {
    ("phase1", 1, 1): "7b51b2b62645e2c6",
    ("phase1", 2, 1): "451157be782c4b81",
    ("phase1", 4, 2): "8346151a30c07824",
    ("composite", 1, 1): "467d3d83cf5a1ae6",
    ("composite", 2, 1): "6c114a20a951ebe0",
    ("composite", 4, 2): "29f8fea8d8af622e",
    ("scale6", 1, 1): "99dbb25f5c8229c9",
    ("scale6", 2, 1): "a973457c1c154c52",
    ("scale6", 4, 2): "3d4426bc09a44dfb",
    ("shocked", 1, 1): "b4df7829ce6c9e7e",
    ("shocked", 2, 1): "7a3da6b5c131f9f6",
    ("shocked", 4, 2): "5ae2ee99773d49ae",
}


def _named_config(name: str) -> Configuration:
    return {
        "phase1": lambda: load_config(PHASE1),
        "composite": lambda: load_config(COMPOSITE),
        "scale6": lambda: build_scale_config(6, depth=3),
        "shocked": _shocked,
    }[name]()


def test_default_decode_keeps_every_legacy_fingerprint() -> None:
    """Unset or explicit `logits-v1` gives the base commit's fingerprint byte for byte (the field
    is omitted at its default, as `producer_obs_version` v1 and the families' capacity_scale)."""
    configs: dict[str, Configuration] = {}
    for (name, o, p), want in BASE_FINGERPRINTS.items():
        cfg = configs.setdefault(name, _named_config(name))
        assert config_fingerprint(cfg, o, p) == want
        assert config_fingerprint(cfg, o, p, ordering_decode=LOGITS_V1) == want


def test_fingerprint_changes_with_ordering_decode() -> None:
    seen: set[str] = set()
    for (name, o, p), legacy in BASE_FINGERPRINTS.items():
        choice = config_fingerprint(_named_config(name), o, p, ordering_decode=CHOICE_V1)
        assert choice != legacy and choice not in seen
        seen.add(choice)
    with pytest.raises(ValueError, match="unknown ordering_decode"):
        config_fingerprint(load_config(COMPOSITE), ordering_decode="choice-v2")


def test_decode_pins_differ_only_off_default() -> None:
    """A driver's pins merge `decode_pins`: unchanged for `logits-v1`, a new key for `choice-v1`,
    so a choice run's spec.json pins (and resume check) never equal a logits run's."""
    pins = {"horizon": 100, "grid": ["w"]}
    assert pins | decode_pins(LOGITS_V1) == pins
    assert pins | decode_pins(CHOICE_V1) != pins
    assert decode_pins(CHOICE_V1) == {"ordering_decode": CHOICE_V1}
    with pytest.raises(ValueError):
        decode_pins("logits-v2")


def test_env_records_decode_and_default_spaces_are_legacy() -> None:
    legacy = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=HORIZON)
    assert legacy.ordering_decode == LOGITS_V1 == legacy.spec_.ordering_decode
    agent = legacy.possible_agents[-1]
    space = legacy.action_space(agent)
    assert type(space).__name__ == "Box" and not isinstance(space, ChoiceShareSpace)
    assert float(space.low.min()) == -ACTION_LOGIT_BOUND
    choice = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=HORIZON,
                              ordering_decode=CHOICE_V1)
    cs = choice.action_space(agent)
    assert isinstance(cs, ChoiceShareSpace) and cs.shape == space.shape
    assert (float(cs.low.min()), float(cs.high.max())) == (0.0, 1.0)
    # producing spaces never change
    assert choice.action_space(choice.possible_agents[0]) == legacy.action_space(
        legacy.possible_agents[0])
    with pytest.raises(ValueError, match="unknown ordering_decode"):
        TokenomicsAECEnv(config=load_config(COMPOSITE), ordering_decode="softmax")


# --- validity layer (unit) ---


def test_whole_choice_is_one_sub_order_with_everything() -> None:
    """Spec "Whole choice": a one-hot share on p sends exactly one sub-order of quantity q with all
    of the order's tokens to p, at any q."""
    ids = _ids(5)
    mask = np.array([True, False, True, True, False])
    for q, tokens in ((1, 0), (1, 7), (13, 1000), (997, 3), (10_000, 123_457)):
        for p in np.flatnonzero(mask):
            shares = np.zeros(5, dtype=np.float32)
            shares[p] = 1.0
            (sub,) = validate_choice(ids, shares, mask, q, tokens)
            assert (sub.producer_id, sub.quantity, sub.tokens) == (ids[p], q, tokens)


@pytest.mark.parametrize("shares, match", [
    ([0.0, 1.0, 0.0, 0.0, 0.0], "ineligible producer p1"),
    ([0.5, 0.25, 0.25, 0.0, 0.0], "ineligible producer p1"),
    ([0.0, 0.0, 1e-9, 0.0, 0.0], "sum"),  # eligible-only support, not summing to one
    ([1.0, 0.0, -0.0, 0.0, 1e-7], "ineligible producer p4"),
    ([1.2, 0.0, -0.2, 0.0, 0.0], "negative share"),
    ([np.nan, 0.0, 1.0, 0.0, 0.0], "non-finite"),
    ([0.5, 0.0, 0.5 + 2 * SHARE_SUM_TOL, 0.0, 0.0], "sum"),
    ([0.5, 0.0, 0.5 - 2 * SHARE_SUM_TOL, 0.0, 0.0], "sum"),
    ([0.0, 0.0, 0.0, 0.0, 0.0], "sum"),
])
def test_invalid_choice_raises_never_repairs(shares: list[float], match: str) -> None:
    """Spec "Ineligible producer chosen" (and the other validity rules): the layer raises."""
    mask = np.array([True, False, True, True, False])
    with pytest.raises(InvalidChoice, match=match):
        validate_choice(_ids(5), np.asarray(shares), mask, 10, 100)


def test_sum_tolerance_edge_is_valid_and_conserves() -> None:
    mask = np.array([True, True, True])
    for eps in (SHARE_SUM_TOL * 0.99, -SHARE_SUM_TOL * 0.99):
        subs = validate_choice(_ids(3), np.array([0.5, 0.25, 0.25 + eps]), mask, 10**7, 10**9)
        assert sum(s.quantity for s in subs) == 10**7 and sum(s.tokens for s in subs) == 10**9


def test_no_eligible_producer_routes_nothing_only_on_a_zero_row() -> None:
    mask = np.zeros(3, dtype=bool)
    assert validate_choice(_ids(3), np.zeros(3), mask, 5, 50) == []
    with pytest.raises(InvalidChoice):
        validate_choice(_ids(3), np.array([1.0, 0.0, 0.0]), mask, 5, 50)


def test_conservation_under_bursts_property() -> None:
    """Spec "Conservation under bursts": any valid share vector, at every q up to 10x the world's
    max consumer quantity (bursts are 5-10x), stays inside the action space and decodes to
    sub-orders that conserve quantity and tokens exactly, on eligible producers only, each
    within one unit of shares x q. Seeded property loop (no hypothesis dependency)."""
    cfg = _shocked()
    env = TokenomicsAECEnv(config=cfg, horizon=HORIZON, ordering_decode=CHOICE_V1)
    spec = env.spec_
    space = env.action_space(env.possible_agents[-1])
    ids, P, K = spec.producer_ids, spec.num_producers, spec.k_stream
    q_cap = 10 * max(c.quantity_max for c in cfg.consumers)
    assert q_cap >= 10
    declares = {p.id: p.produces for p in cfg.producer_configs}
    products = sorted({prod for ps in declares.values() for prod in ps})
    rng = np.random.default_rng(20261030)
    whole = split = 0
    qs = list(range(1, q_cap + 1)) + [int(x) for x in rng.integers(1, q_cap + 1, size=1500)]
    for trial, q in enumerate(qs):
        product = products[trial % len(products)]
        mask = np.array([product in declares[pid] for pid in ids])
        elig = np.flatnonzero(mask)
        assert elig.size
        shares = np.zeros(P, dtype=np.float32)
        kind = trial % 4
        if kind == 0:  # whole
            shares[rng.choice(elig)] = 1.0
        else:  # a split over a random eligible support, float32 as a policy would emit it
            support = rng.choice(elig, size=int(rng.integers(1, elig.size + 1)), replace=False)
            w = rng.dirichlet(np.full(support.size, 0.3 if kind == 3 else 1.0))
            shares[support] = w.astype(np.float32)
            shares /= np.float32(shares.sum(dtype=np.float32))
        action = np.zeros((K, P), dtype=np.float32)
        action[int(rng.integers(K))] = shares
        assert space.contains(action.reshape(-1))
        tokens = int(rng.integers(0, 50 * q + 1))
        subs = validate_choice(ids, shares, mask, q, tokens)
        assert sum(s.quantity for s in subs) == q
        assert sum(s.tokens for s in subs) == tokens
        assert all(s.quantity > 0 and s.tokens >= 0 for s in subs)
        assert len({s.producer_id for s in subs}) == len(subs)
        for s in subs:
            i = ids.index(s.producer_id)
            assert mask[i] and shares[i] > 0
            assert abs(s.quantity - float(shares[i]) * q) < 1.0 + 1e-9
        if np.count_nonzero(shares) == 1:
            whole += 1
            assert len(subs) == 1 and subs[0].tokens == tokens
        else:
            split += 1
    assert whole > 300 and split > 300


def test_choice_quantities_follow_the_legacy_largest_remainder_rule() -> None:
    """Same apportioning as the legacy decode: shares equal to the softmax of legacy logits give
    the same sub-orders as `allocate_suborders`."""
    rng = np.random.default_rng(5)
    ids = _ids(6)
    mask = np.array([True, True, False, True, True, False])
    for _ in range(200):
        logits = rng.normal(size=6)
        z = np.exp(logits[mask] - logits[mask].max())
        shares = np.zeros(6)
        shares[mask] = z / z.sum()
        q, t = int(rng.integers(1, 200)), int(rng.integers(0, 5000))
        assert validate_choice(ids, shares, mask, q, t) == allocate_suborders(ids, logits, mask, q, t)
        assert [s.quantity for s in validate_choice(ids, shares, mask, q, t)] == [
            x for x in _allocate_int(shares[mask] / shares[mask].sum(), q) if x > 0]


# --- decode_ordering under choice-v1 ---


def _view(n_orders: int, products: list[str], eligible_ids: dict[str, list[str]]) -> OrderingAgentView:
    incoming = {f"o{i:02d}": OrderInfo(id=f"o{i:02d}", tokens=10 * (i + 1),
                                       product=products[i % len(products)], quantity=3 + i)
                for i in range(n_orders)}
    eligible = {
        oid: {pid: ProducerInfo(id=pid, capacities={}, max_capacities={}, cut_off_price=0.0,
                                produces=frozenset({oi.product}))
              for pid in eligible_ids[oi.product]}
        for oid, oi in incoming.items()}
    return OrderingAgentView(incoming, eligible)


def _choice_spec(k: int) -> EnvSpec:
    """A choice-v1 spec with a forced k (legacy obs: k_stream == k)."""
    spec = build_spec(load_config(COMPOSITE), ordering_decode=CHOICE_V1)
    return replace(spec, k=k)


def test_overflow_orders_keep_the_legacy_even_split() -> None:
    """Orders beyond `k_stream` have no slot: legacy even split over their eligible producers,
    identical to the logits decode's overflow; slotted orders follow their shares."""
    spec = _choice_spec(2)
    ids = spec.producer_ids
    assert len(ids) >= 3
    prods = ["a", "b"]
    elig = {"a": [ids[0], ids[1]], "b": [ids[1], ids[2]]}
    view = _view(5, prods, elig)
    action = np.zeros((2, len(ids)), dtype=np.float32)
    action[0, ids.index(ids[1])] = 1.0  # o00 (product a) whole to ids[1]
    action[1, ids.index(ids[2])] = 1.0  # o01 (product b) whole to ids[2]
    cmd = decode_ordering(spec, action.reshape(-1), view)
    legacy = decode_ordering(replace(spec, ordering_decode=LOGITS_V1),
                             np.zeros(2 * len(ids)), view)
    assert [(s.producer_id, s.quantity) for s in cmd.orders["o00"]] == [(ids[1], 3)]
    assert [(s.producer_id, s.quantity) for s in cmd.orders["o01"]] == [(ids[2], 4)]
    for oid in ("o02", "o03", "o04"):  # overflow
        assert cmd.orders[oid] == legacy.orders[oid]
        assert len(cmd.orders[oid]) == 2  # even split, not a choice


def test_empty_slot_must_be_zero_and_slot_errors_name_the_order() -> None:
    spec = _choice_spec(3)
    ids = spec.producer_ids
    view = _view(1, ["a"], {"a": [ids[0]]})
    action = np.zeros((3, len(ids)), dtype=np.float32)
    action[0, 0] = 1.0
    assert isinstance(decode_ordering(spec, action.reshape(-1), view), OrderingAgentCommand)
    stray = action.copy()
    stray[2, 1] = 1.0
    with pytest.raises(InvalidChoice, match="slot 2 has no order"):
        decode_ordering(spec, stray.reshape(-1), view)
    bad = np.zeros_like(action)
    bad[0, 1] = 1.0
    with pytest.raises(InvalidChoice, match=r"order o00 \(slot 0\).*ineligible"):
        decode_ordering(spec, bad.reshape(-1), view)


def test_share_space_sample_with_mask_is_a_valid_whole_choice() -> None:
    space = ChoiceShareSpace(3, 4)
    space.seed(0)
    mask = np.array([[0, 1, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int8)
    for _ in range(50):
        a = space.sample(mask.reshape(-1))
        assert space.contains(a)
        rows = a.reshape(3, 4)
        assert rows[0].sum() == 1.0 and rows[0][mask[0] == 0].sum() == 0
        assert rows[1].sum() == 0.0
        assert rows[2].tolist() == [1.0, 0.0, 0.0, 0.0]


# --- env level ---


def _mask_policy(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    """A whole choice per slot drawn from the env's `action_mask` info (choice-v1)."""
    return env.action_space(agent).sample(env.infos[agent]["action_mask"])


@pytest.mark.parametrize("decode", [LOGITS_V1, CHOICE_V1])
@pytest.mark.parametrize("world", ["composite", "joint_stack"])
def test_pettingzoo_api_on_both_decodes(decode: str, world: str) -> None:
    if world == "composite":
        env = TokenomicsAECEnv(config=load_config(COMPOSITE), horizon=40, ordering_decode=decode)
    else:
        env = joint_stack.joint_stack_env(_calm(), HORIZON, ordering_decode=decode)
    api_test(env, num_cycles=20, verbose_progress=False)


def test_action_mask_info_only_under_choice() -> None:
    for decode in (LOGITS_V1, CHOICE_V1):
        env = joint_stack.joint_stack_env(_calm(), HORIZON, ordering_decode=decode)
        env.reset(seed=1)
        masks = 0
        for agent in env.agent_iter(max_iter=len(env.possible_agents) * 2):
            obs, _, term, trunc, info = env.last()
            role = env._role[agent][0]
            if decode == LOGITS_V1 or role == "prod":
                assert "action_mask" not in info
            else:
                assert info["action_mask"].shape == env.action_space(agent).shape
                masks += 1
            act = (PRODUCING_BASELINES["entity_planner"](env, agent) if role == "prod"
                   else (_mask_policy(env, agent) if decode == CHOICE_V1
                         else ORDERING_BASELINES["even_split"](env, agent)))
            env.step(act)
        assert masks > 0 if decode == CHOICE_V1 else masks == 0


def test_whole_choice_episode_routes_every_slotted_order_whole(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec "Whole choice" end to end: a horizon-12 shocked 12-producer episode under choice-v1
    with whole choices; every slotted order goes to exactly one eligible producer with its whole
    quantity and every token, and the domain accepts every command."""
    seen: list[tuple[OrderingAgentView, OrderingAgentCommand, int]] = []
    real = aec_env.decode_ordering

    def spy(spec, action, view):  # noqa: ANN001, ANN202 - test wrapper
        cmd = real(spec, action, view)
        seen.append((view, cmd, spec.k_stream))
        return cmd

    monkeypatch.setattr(aec_env, "decode_ordering", spy)
    env = joint_stack.joint_stack_env(_shocked(), HORIZON, ordering_decode=CHOICE_V1)
    stats = run_episode(env, PRODUCING_BASELINES["entity_planner"], _mask_policy, seed=3,
                        action_seed=3)
    assert stats["cycles"] == HORIZON
    slotted = 0
    for view, cmd, k_stream in seen:
        for k, oid in enumerate(sorted(view.incoming)):
            if k >= k_stream:
                continue
            oi = view.incoming[oid]
            (sub,) = cmd.orders[oid]
            assert sub.producer_id in view.eligible[oid]
            assert (sub.quantity, sub.tokens) == (oi.quantity, oi.tokens)
            slotted += 1
    assert slotted > 50


def test_choice_env_is_deterministic_given_seeds() -> None:
    def run() -> dict:
        env = joint_stack.joint_stack_env(_shocked(), HORIZON, ordering_decode=CHOICE_V1)
        return run_episode(env, PRODUCING_BASELINES["entity_planner"], _mask_policy, seed=4,
                           action_seed=9)

    assert run() == run()


# --- contract path ---


BAD = "choice_bad"
GOOD = "choice_whole"


def _bad_choice(env: TokenomicsAECEnv, agent: str) -> np.ndarray:
    """Whole choices, except the first occupied slot's share goes to an ineligible producer."""
    mask = env.infos[agent]["action_mask"].reshape(env.spec_.k_stream, env.spec_.num_producers)
    action = env.action_space(agent).sample(mask.reshape(-1)).reshape(mask.shape)
    for k in range(mask.shape[0]):
        if mask[k].any() and not mask[k].all():
            action[k] = 0.0
            action[k, int(np.flatnonzero(mask[k] == 0)[0])] = 1.0
            break
    return action.reshape(-1)


@pytest.fixture(scope="module")
def gate_inputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One calm world at capacity scale 0.1 and a one-member LEGACY roster (`noop/obs_price`),
    so only the candidate plays the joint-stack env the test switches to choice-v1."""
    d = tmp_path_factory.mktemp("gate_inputs")
    full = build_worlds_v1(SEED_ORDER)
    entry = scaled_entry(full.entry(CALM), 0.1)
    save_registry(Registry(name="fixture", seed_ranges=full.seed_ranges, entries=(entry,)),
                  d / "registry.json")
    save_roster(Roster(name="fixture-roster", replicates=1, members=(
        RosterMember("noop", "obs_price", contract_for("noop", "obs_price"), False),)),
        d / "roster.json")
    (d / "params.json").write_text(side_effect_params_json(0.5, 0.0, 0.05, name="fixture"))
    return {"registry": d / "registry.json", "roster": d / "roster.json",
            "params": d / "params.json"}


def _gate(inputs: dict[str, Path], out: Path, ordering: str, monkeypatch: pytest.MonkeyPatch) -> int:
    real = joint_stack.joint_stack_env
    with monkeypatch.context() as m:
        m.setitem(ORDERING_BASELINES, ordering, {BAD: _bad_choice, GOOD: _mask_policy}[ordering])
        m.setattr(joint_stack, "joint_stack_env",
                  lambda cfg, horizon: real(cfg, horizon, ordering_decode=CHOICE_V1))
        return main(["--mode", "report", "--out-dir", str(out),
                     "--subject", f"baseline:entity_planner/{ordering}",
                     "--registry", str(inputs["registry"]), "--roster", str(inputs["roster"]),
                     "--params", str(inputs["params"]), "--horizon", str(HORIZON),
                     "--episodes-per-world", "1", "--workers", "1", "--dry-run"])


def _failures(out: Path) -> list[dict[str, str]]:
    with (out / "contract_failures.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_ineligible_choice_is_a_job_contract_failure(gate_inputs: dict[str, Path],
                                                     tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec "Ineligible producer chosen" through the gate's normal job path: the action is inside
    its Box (the monitor passes it), the validity layer raises in `env.step`, the job's
    `contract_failures.csv` row carries check `choice_validity`, and the run is invalid (exit 2).
    Nothing is rerouted: the candidate has no episode row."""
    out = tmp_path / "bad"
    assert _gate(gate_inputs, out, BAD, monkeypatch) == 2
    (f,) = _failures(out)
    assert tuple(f) == FAILURE_COLUMNS
    assert (f["scope"], f["check"], f["error_type"], f["world"], f["subject"]) == (
        "job", CHOICE, "InvalidChoice", CALM, "candidate")
    assert "ineligible producer" in f["message"]
    assert [r.subject for r in read_rows(out / "episodes.csv")] == ["noop/obs_price"]
    spec = json.loads((out / "spec.json").read_text())
    assert spec["status"] == "invalid"


def test_valid_whole_choices_run_clean_through_the_gate(gate_inputs: dict[str, Path],
                                                        tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control for the test above: the same path with valid whole choices records no
    contract failure."""
    out = tmp_path / "good"
    code = _gate(gate_inputs, out, GOOD, monkeypatch)
    assert _failures(out) == []
    assert sorted(r.subject for r in read_rows(out / "episodes.csv")) == [
        "candidate", "noop/obs_price"]
    assert code != 2


def test_job_failure_keeps_the_choice_check_name() -> None:
    job = {"world": "w", "episode": 1, "subject": "candidate", "replicate": 0}
    f = job_failure(job, InvalidChoice("share 1.0 on ineligible producer p3"))
    assert (f.scope, f.check, f.error_type) == ("job", CHOICE, "InvalidChoice")
    assert f.message == "share 1.0 on ineligible producer p3"
