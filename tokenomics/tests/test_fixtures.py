"""Out-of-family / scarcity / degenerate fixture gates (PRODUCT_AGNOSTIC step 5).

These fixtures are the C3 transfer targets: economies whose RECIPE DAG leaves the
layered single-input scale family (`build_scale_config`). A fixture is only useful if
it (a) validates as a real economy and (b) is genuinely out-of-family — otherwise a
"transfer" row proves nothing. These tests pin both before the matrix consumes them.
"""

from __future__ import annotations

from tokenomics.eval.fixtures import build_forkjoin_config
from tokenomics.eval.scale import build_scale_config


def _max_inputs(cfg) -> int:
    return max((len(ps.inputs) for ps in cfg.process_sheets), default=0)


def test_forkjoin_fixture_is_valid_and_out_of_family() -> None:
    # The scale family is single-input (every composite product consumes exactly one
    # lower product). A fork/join fixture must carry at least one MULTI-input product —
    # that is what makes it out-of-family, not merely a re-seeded scale economy.
    assert _max_inputs(build_scale_config(0, depth=3)) == 1

    cfg = build_forkjoin_config(0, depth=3, width=4, inputs_per_product=2)
    cfg.validate()  # must be a real, feasible economy (raises otherwise)
    assert _max_inputs(cfg) == 2  # genuinely out-of-family: join of 2 inputs
