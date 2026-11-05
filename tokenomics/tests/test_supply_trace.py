"""Supply observer (add-m4-orderer-redesign task 7.14): planted losses per class, the consistency
check, and (with the EXP-0052 / EXP-0057 records) a 1-world dry run and EXP-0057 neutrality."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tokenomics.domain.config import ProcessSheet, ProducingAgentConfig, Restoration, Upgrade
from tokenomics.domain.order import Order
from tokenomics.domain.producing_agent import ProducingAgent
from tokenomics.domain.requests import ConsumerRequest, ReplenishmentRequest
from tokenomics.domain.system import BidOutcome, OrderResolution
from tokenomics.eval import supply_trace as st

REPO = Path(__file__).resolve().parents[2]
ATLAS_V2 = REPO / "experiments/EXP-0052-headroom-atlas-confirm/reclassified/atlas_v2.json"
EXP0057 = REPO / "experiments/EXP-0057-decoder-check/artifacts/episodes.csv"
ORDERER = REPO / "experiments/EXP-0012-portable-orderer-scorer/artifacts/orderer_scorer.pt"
WORLD = "ws-atlas-v1-12-0060"

GOOD = ProcessSheet("good", {"a": 2}, {"raw": 1})  # composite: 2 a + 1 raw per unit
RAW = ProcessSheet("raw", {"u": 1})                # primal, made upstream
CONSUMER, REPL = 1, 2


def _producer(pid: str, caps: dict[str, int], sheet: ProcessSheet) -> ProducingAgent:
    t = next(iter(caps))
    cfg = ProducingAgentConfig(id=pid, capacities=caps, degradation=0,
                               restoration=Restoration("raw", 1, t), upgrade=Upgrade("raw", 1, t),
                               produces=frozenset({sheet.product}), inventory_limit=10)
    return ProducingAgent(cfg, {sheet.product: sheet})


class World:
    """A fake system: P makes `good` from `raw`, U makes `raw`; the observer is driven by planted
    bid outcomes and resolutions, as `System.complete_cycle` would hand them over."""

    def __init__(self, p_caps: int = 10, p_max: int = 10, u_caps: int = 5) -> None:
        self.p = _producer("P", {"a": p_caps}, GOOD)
        self.p.max_capacities["a"] = p_max
        self.u = _producer("U", {"u": u_caps}, RAW)
        self.system = SimpleNamespace(cycle_counter=0, _outages_by_cycle={},
                                      producing_agents={"P": self.p, "U": self.u}, orders={},
                                      complete_cycle=lambda: None)
        self.obs = st.SupplyObserver()
        self.obs.attach_system(self.system)
        self.system.orders[CONSUMER] = Order.new_consumer_order(
            CONSUMER, GOOD, ConsumerRequest(consumer_id="c0", product="good", tokens=10,
                                            quantity=2))

    def add_replenishment(self) -> None:
        self.system.orders[REPL] = Order.new_replenishment_order(
            REPL, RAW, ReplenishmentRequest("P", "raw", 2, float("nan")))

    def cycle(self, outcomes: list[tuple], resolutions: list[tuple], outage: bool = False) -> None:
        c = self.system.cycle_counter = self.system.cycle_counter + 1
        if outage:
            self.system._outages_by_cycle[c] = {"P"}
        self.obs.before(self.system)
        result = SimpleNamespace(
            bid_outcomes=[BidOutcome(pid, oid, kind, prod, 2, 4, out, reason)
                          for pid, oid, kind, prod, out, reason in outcomes],
            order_resolutions=[OrderResolution(oid, kind, prod, 2, done, 0)
                               for oid, kind, prod, done in resolutions])
        self.obs.after(result)
        for oid, *_ in resolutions:
            self.system.orders.pop(oid, None)

    def classes(self) -> dict[str, float]:
        cells = self.obs.cells()
        return {k: float(cells[f"st_{k}"]) for k in st.CLASSES if float(cells[f"st_{k}"])}


def _lose(world: World, reason: str, outcome: str = "rejected", outage: bool = False) -> None:
    world.cycle([("P", CONSUMER, "consumer", "good", outcome, reason)],
                [(CONSUMER, "consumer", "good", False)], outage=outage)


@pytest.mark.parametrize("p_caps,p_max,cls", [(10, 10, st.A_HELD), (3, 10, st.A_DEGRADED),
                                              (3, 3, st.A_MAX)])
def test_no_fit_splits_by_capacity(p_caps: int, p_max: int, cls: str) -> None:
    w = World(p_caps=p_caps, p_max=p_max)
    _lose(w, "no_fit")
    assert w.classes() == {cls: 2.0}


@pytest.mark.parametrize("reason,outcome,outage,cls", [
    ("dropped", "failed", False, st.A_DROPPED), ("displaced", "rejected", False, st.C_DISPLACED),
    ("outage", "rejected", True, st.D_OUTAGE), ("ttl", "failed", True, st.D_TTL_OUTAGE)])
def test_reason_classes(reason: str, outcome: str, outage: bool, cls: str) -> None:
    w = World()
    _lose(w, reason, outcome, outage)
    assert w.classes() == {cls: 2.0}


def test_never_bid() -> None:
    w = World()
    w.cycle([], [(CONSUMER, "consumer", "good", False)])
    assert w.classes() == {st.NEVER_BID: 2.0}


def test_inputs_in_stock_is_contention() -> None:
    w = World()
    w.p.inventory["raw"] = 2
    _lose(w, "ttl", "failed")
    assert w.classes() == {st.B4: 2.0}


def test_upstream_unable_is_b3() -> None:
    w = World(u_caps=0)
    _lose(w, "ttl", "failed")
    assert w.classes() == {st.B3: 2.0}


def test_no_input_order_is_b1() -> None:
    w = World()
    _lose(w, "ttl", "failed")
    assert w.classes() == {st.B1: 2.0}


def _accept_then(w: World, repl_outcomes: list[tuple], repl_res: list[tuple]) -> None:
    w.add_replenishment()
    w.cycle([("P", CONSUMER, "consumer", "good", "accepted", None)] + repl_outcomes, repl_res)


def test_input_delivered_in_the_failing_cycle_is_late() -> None:
    w = World()
    _accept_then(w, [], [])
    w.cycle([("P", CONSUMER, "consumer", "good", "failed", "ttl")],
            [(REPL, "replenishment", "raw", True), (CONSUMER, "consumer", "good", False)])
    assert w.classes() == {"b2_late": 2.0}


def test_input_delivered_earlier_but_short() -> None:
    w = World()
    _accept_then(w, [], [(REPL, "replenishment", "raw", True)])
    w.cycle([("P", CONSUMER, "consumer", "good", "failed", "ttl")],
            [(CONSUMER, "consumer", "good", False)])
    assert w.classes() == {"b2_short": 2.0}


def test_unfunded_input_order() -> None:
    w = World()
    _accept_then(w, [], [(REPL, "replenishment", "raw", False)])  # expired without a bid
    w.cycle([("P", CONSUMER, "consumer", "good", "failed", "ttl")],
            [(CONSUMER, "consumer", "good", False)])
    assert w.classes() == {"b2_unfunded": 2.0}


def test_input_order_lost_upstream_carries_its_reason() -> None:
    w = World()
    _accept_then(w, [("U", REPL, "replenishment", "raw", "rejected", "no_fit")],
                 [(REPL, "replenishment", "raw", False)])
    w.cycle([("P", CONSUMER, "consumer", "good", "failed", "ttl")],
            [(CONSUMER, "consumer", "good", False)])
    assert w.classes() == {"b2_lost_no_fit": 2.0}


def test_in_flight_input_order_resolves_later_or_stays_open() -> None:
    w = World()
    _accept_then(w, [], [])
    w.cycle([("P", CONSUMER, "consumer", "good", "failed", "ttl")],
            [(CONSUMER, "consumer", "good", False)])
    assert w.classes() == {"b2_open": 2.0}  # in flight at the episode's end
    w.cycle([], [(REPL, "replenishment", "raw", True)])
    assert w.classes() == {"b2_late": 2.0}  # delivered after the failure


def test_multi_input_precedence_upstream_first() -> None:
    sheet = ProcessSheet("good", {"a": 2}, {"raw": 1, "rare": 1})
    w = World()
    w.p.sheets["good"] = sheet
    rare = _producer("R", {"r": 0}, ProcessSheet("rare", {"r": 1}))
    w.system.producing_agents["R"] = rare
    w.obs.decl["rare"].append("R")
    _lose(w, "ttl", "failed")  # raw: b1 (U can make it, not ordered); rare: b3 (R cannot)
    assert w.classes() == {st.B3: 2.0}
    assert float(w.obs.cells()["st_multi_missing"]) == 2.0


def test_producer_cycles_record_capacity_stock_and_orders() -> None:
    w = World(p_caps=5, p_max=10)
    w.p.inventory["raw"] = 3
    w.add_replenishment()
    w.cycle([], [(REPL, "replenishment", "raw", True)])
    rows = {r[1]: dict(zip(st.PC_FIELDS, r)) for r in json.loads(w.obs.cells()[st.PC_CELL])}
    assert rows["P"]["capacity_ratio"] == 0.5 and rows["P"]["input_stock_units"] == 3
    assert rows["P"]["replenishment_units_placed"] == 2
    assert rows["P"]["replenishment_units_delivered"] == 2


def _row(**cells: float) -> dict[str, str]:
    d = {"world": "w", "episode": "1", "subject": "s", "replicate": "0"}
    for r in st.LOSS_REASONS:
        d[f"trace_lost_units_consumer_{r}"] = "0.0"
        d[f"st_reason_{r}"] = "0.0"
    for k in st.CLASSES + st.CHECK_COUNTERS:
        d[f"st_{k}"] = "0.0"
    d["st_lost_consumer_total"] = "0.0"
    d.update({k: repr(float(v)) for k, v in cells.items()})
    return d


def test_consistency_flags_classes_off_the_traced_loss() -> None:
    ok = _row(trace_lost_units_consumer_ttl=5, st_reason_ttl=5, st_b1_not_ordered=2,
              st_b2_late=3, st_lost_consumer_total=5)
    assert st.consistency([ok]) == []
    off = dict(ok, st_b2_late="2.0")
    assert [c for c, _ in st.consistency([off])] == [st.CONSISTENCY]
    counter = dict(ok, st_history_missing="1.0")
    assert st.consistency([counter])


def test_reading_largest_and_tie() -> None:
    rows = [dict(_row(st_b1_not_ordered=50, st_b2_late=46, st_a_held=4), size="12", units="1",
                 world="w12"),
            dict(_row(st_c_displaced=10), size="42", units="1", world="w42")]
    r = st.supply_reading(rows, [12, 42])
    assert r["reading"] == "tie" and r["tie_with"] == ["b2_not_delivered"]
    rows[0] = dict(rows[0], st_b2_late="30.0")
    r = st.supply_reading(rows, [12, 42])
    assert r["reading"] == "b1_not_ordered" and r["per_size"][42]["largest"] == "c_outbid"


def _lfs(p: Path) -> bool:
    return p.exists() and not p.read_bytes()[:40].startswith(b"version https://git-lfs")


@pytest.mark.skipif(not all(_lfs(p) for p in (ATLAS_V2, EXP0057, ORDERER)),
                    reason="EXP-0052 / EXP-0057 / EXP-0012 records absent")
def test_one_world_dry_run_neutral_and_consistent(tmp_path: Path) -> None:
    """Horizon 100 on one world (4 MWWk rows, ~10 s): every row equals EXP-0057's, the classes
    sum to the traced loss, and the outputs exist."""
    out = tmp_path / "EXP-dry"
    assert st.main(["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER),
                    "--exp0057-episodes", str(EXP0057), "--dry-run", "--worlds", WORLD,
                    "--workers", "1", "--out-dir", str(out)]) == 0
    neutral = list(csv.DictReader((out / "neutrality.csv").open()))
    assert len(neutral) == 4 and all(r["equal"] == "true" for r in neutral)
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == "valid" and reading["neutral_rows"] == 4
    assert (out / "producer_cycles.csv").exists() and (out / "supply_trace.csv").exists()


def test_real_run_refuses_filters(tmp_path: Path) -> None:
    if not all(_lfs(p) for p in (ATLAS_V2, ORDERER)):
        pytest.skip("records absent")
    with pytest.raises(ValueError, match="not a valid supply-trace run"):
        st.run_supply_trace(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt=str(ORDERER),
                            worlds=[WORLD], source_commit="0" * 40)


# --- task 7.17a: the TTL-sweep profile (both arms, --order-ttl) ---

EXP0073_TTL2 = REPO / "experiments/EXP-0073-ttl-sweep/ttl2/artifacts/episodes.csv"
needs_records = pytest.mark.skipif(
    not all(_lfs(p) for p in (ATLAS_V2, EXP0057, ORDERER, EXP0073_TTL2)),
    reason="EXP-0052 / EXP-0057 / EXP-0073 / EXP-0012 records absent")


def _args(out: Path, *extra: str) -> list[str]:
    return ["--atlas", str(ATLAS_V2), "--orderer-ckpt", str(ORDERER), "--dry-run", "--worlds",
            WORLD, "--workers", "1", "--out-dir", str(out), *extra]


@needs_records
def test_sweep_refusals(tmp_path: Path) -> None:
    kw = dict(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt=str(ORDERER))
    with pytest.raises(ValueError, match="arms within"):
        st.run_supply_trace(**kw, order_ttl=2, arms=["TR"], dry_run=True)
    with pytest.raises(ValueError, match="TTL 2 only"):
        st.run_supply_trace(**kw, order_ttl=4, exp0073_episodes=EXP0073_TTL2, dry_run=True)
    with pytest.raises(ValueError, match="belong to the TTL sweep"):
        st.run_supply_trace(**kw, arms=["RR"], dry_run=True)
    with pytest.raises(ValueError, match="EXP-0073 ttl2"):  # a real TTL-2 sweep needs both refs
        st.run_supply_trace(**kw, order_ttl=2, exp0057_episodes=EXP0057, source_commit="0" * 40)


@needs_records
def test_sweep_ttl2_equals_exp0057_and_exp0073(tmp_path: Path) -> None:
    """TTL 2, horizon 100, one world, both arms (6 rows): every row equals EXP-0057's and
    EXP-0073 ttl2's; TTL pinned; the 7.15 shares reported per arm."""
    out = tmp_path / "ttl2"
    assert st.main(_args(out, "--order-ttl", "2", "--exp0057-episodes", str(EXP0057),
                         "--exp0073-episodes", str(EXP0073_TTL2))) == 0
    for f in ("neutrality.csv", "neutrality_exp0073.csv"):
        rows = list(csv.DictReader((out / f).open()))
        assert len(rows) == 6 and all(r["equal"] == "true" for r in rows), f
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["order_ttl"] == 2 and spec["pins"]["profile"] == "ttl_sweep"
    reading = json.loads((out / "reading.json").read_text())
    assert reading["status"] == "valid" and set(reading["per_arm"]) == {"MWWk", "RR"}
    assert {r["arm"] for r in csv.DictReader((out / "supply_trace.csv").open())} == {"MWWk", "RR"}


@needs_records
def test_sweep_ttl4_dry_run(tmp_path: Path) -> None:
    out = tmp_path / "ttl4"
    assert st.main(_args(out, "--order-ttl", "4", "--horizon", "12")) == 0
    spec = json.loads((out / "spec.json").read_text())
    assert spec["pins"]["order_ttl"] == 4 and spec["result"]["contract_failures"] == 0
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
    assert len(list(csv.DictReader((out / "supply_trace.csv").open()))) == 6


def test_large_producer_cycle_cells_read_without_a_reference(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch
                                                             ) -> None:
    """Task 7.21 (EXP-0077 crash): a run that loads no reference must still read rows whose
    `st_producer_cycles` cell exceeds csv's default 131,072-character limit. In a fresh process
    the default limit refuses such a cell; after `allow_large_cells` it reads; and
    `run_supply_trace` calls it before anything else."""
    import subprocess
    import sys

    big = tmp_path / "big.csv"
    big.write_text("a,b\n1," + "x" * 300_000 + "\n")
    code = ("import csv, sys; from tokenomics.eval import supply_trace as st; "
            f"p = {str(big)!r}; "
            "ok = lambda: len(list(csv.DictReader(open(p))))\n"
            "try:\n    ok(); sys.exit(3)\nexcept csv.Error:\n    pass\n"
            "st.allow_large_cells(); sys.exit(0 if ok() == 1 else 4)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    calls = []
    monkeypatch.setattr(st, "allow_large_cells", lambda: calls.append(1))
    with pytest.raises(ValueError):  # an early refusal: the limit is raised before it
        st.run_supply_trace(atlas_path=ATLAS_V2, out_dir=tmp_path / "o", orderer_ckpt="x",
                            shock_profile="bogus")
    assert calls == [1]


EXP0077_K4 = REPO / "experiments/EXP-0077-buffer-sweep/entity_planner_buyahead_k4/artifacts/episodes.csv"


@pytest.mark.skipif(not all(_lfs(p) for p in (ATLAS_V2, ORDERER, EXP0077_K4)),
                    reason="EXP-0052 / EXP-0012 / EXP-0077 records absent")
def test_task_722_cells_are_added_and_existing_cells_unchanged(tmp_path: Path) -> None:
    """Task 7.22: buy-ahead k = 4 on one world at horizon 100 (6 rows): every cell EXP-0077 wrote
    is equal; the new cells exist, the headroom-blocked late units are a part of `b2_late`, and
    the second producer-cycle cell lines up with the first."""
    import csv as _csv

    _csv.field_size_limit(1 << 30)
    out = tmp_path / "k4"
    assert st.main(_args(out, "--order-ttl", "2", "--producer", "entity_planner_buyahead_k4")) == 0
    ref = {(r["world"], r["episode"], r["subject"], r["replicate"]): r
           for r in _csv.DictReader(EXP0077_K4.open()) if r["world"] == WORLD}
    rows = list(_csv.DictReader((out / "episodes.csv").open()))
    assert len(rows) == len(ref) == 6
    for r in rows:
        old = ref[(r["world"], r["episode"], r["subject"], r["replicate"])]
        assert {c: r[c] for c in old} == old
        assert set(r) - set(old) == {st.LATE_HB, st.PCX_CELL}
        assert 0 <= float(r[st.LATE_HB]) <= float(r["st_b2_late"])
        x = json.loads(r[st.PCX_CELL])
        assert [q[:2] for q in x] == [q[:2] for q in json.loads(r[st.PC_CELL])]
    pc = list(_csv.DictReader((out / "producer_cycles.csv").open()))
    assert set(st.PCX_FIELDS) <= set(pc[0])


@pytest.mark.skipif(not all(_lfs(p) for p in (ATLAS_V2, ORDERER)), reason="records absent")
def test_rf_producer_runs_through_the_observer(tmp_path: Path) -> None:
    out = tmp_path / "rf"
    assert st.main(_args(out, "--order-ttl", "2", "--producer", "entity_planner_buyahead_k4_rf",
                         "--horizon", "12")) == 0
    assert (out / "contract_failures.csv").read_text().count("\n") == 1
