"""THE exactness-equivalence tests (Golden rule).

Every incremental path is compared against from-scratch recomputation on the
same G^Δ, over random interventions, for both architectures and depths. If
anything here fails, no incremental result may be reported or timed.
"""
import os

import pytest
import torch

from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.eval.exactness import assert_exact
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward
from src.models.model import load_model
from src.paths import CHECKPOINTS, FINBENCH_PROC

TR = ("account", "transfer", "account")
NT = "account"

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt")),
    reason="processed FinBench graph not present",
)


@pytest.fixture(scope="module")
def env():
    set_determinism()
    data, schema = load_graph("finbench")
    return data, schema, GraphState(data)


def _model(arch, L):
    path = os.path.join(CHECKPOINTS, "finbench", f"nodelabel_{arch}_L{L}.pt")
    if not os.path.exists(path):
        pytest.skip(f"checkpoint missing: {path}")
    return load_model(path)


def _random_ivs(schema, gs, kind, n, seed=0):
    from src.explore.sampler import sample_feature_iv, sample_mixed_iv, sample_topo_iv
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        cohort = int(torch.randint(1, 200, (1,), generator=g))
        edits = int(torch.randint(1, 50, (1,), generator=g))
        if kind == "feature":
            yield sample_feature_iv(schema, gs, g, NT, cohort)
        elif kind == "topo":
            yield sample_topo_iv(schema, gs, g, TR, n_add=edits, n_del=edits)
        else:
            yield sample_mixed_iv(schema, gs, g, NT, TR, cohort, edits, edits)


IMPLS = ["v1", "v2"]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
@pytest.mark.parametrize("L", [1, 2, 4])
@pytest.mark.parametrize("kind", ["feature", "topo", "mixed"])
def test_incremental_equals_from_scratch(env, arch, L, kind, impl):
    data, schema, gs = env
    model = _model(arch, L)
    cache = build_cache(model, gs.base_view())
    # the cache itself must reproduce the plain forward
    assert_exact(cache.h_final, whatif_forward(model, gs, _empty_iv()))
    for iv in _random_ivs(schema, gs, kind, n=3, seed=hash((arch, L, kind)) % 2**31):
        truth = whatif_forward(model, gs, iv, schema)
        res = incremental_forward(model, gs, cache, iv, schema, impl=impl)
        assert_exact(res.h, truth)
        # the returned cache must be exact state for G^Δ (reuse depends on it)
        assert_exact(res.cache.h_final, truth)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("L", [2, 4])
def test_delta_and_closure_agree(env, L, impl):
    """Feature Δ on RGCN can go through either maintainer; both must match
    truth (and hence each other)."""
    data, schema, gs = env
    model = _model("rgcn", L)
    cache = build_cache(model, gs.base_view())
    for iv in _random_ivs(schema, gs, "feature", n=3, seed=7):
        truth = whatif_forward(model, gs, iv, schema)
        d = incremental_forward(model, gs, cache, iv, schema, method="delta", impl=impl)
        c = incremental_forward(model, gs, cache, iv, schema, method="closure", impl=impl)
        assert d.stats["method"] == "delta" and c.stats["method"] == "closure"
        assert_exact(d.h, truth)
        assert_exact(c.h, truth)


@pytest.mark.parametrize("impl", IMPLS)
def test_delta_within_float32_noise_of_double_reference(env, impl):
    """Guards the loosened atol (see configs/common.yaml): the delta path must
    be no further from the float64 truth than from-scratch float32 is. A real
    delta math bug would exceed float32 accumulation noise and fail here even
    though it might slip under the 1e-4 atol against float32 from-scratch."""
    data, schema, gs = env
    model = _model("rgcn", 4)                       # deepest: worst float headroom
    cache = build_cache(model, gs.base_view())

    # float64 reference model + graph
    model64 = load_model(os.path.join(CHECKPOINTS, "finbench", "nodelabel_rgcn_L4.pt")).double()
    gs64 = GraphState(data)
    for nt in gs64._x:
        gs64._x[nt] = gs64._x[nt].double()

    from src.data.graph_state import Intervention
    for iv in _random_ivs(schema, gs, "feature", n=3, seed=99):
        inc = incremental_forward(model, gs, cache, iv, method="delta", impl=impl).h
        scratch32 = whatif_forward(model, gs, iv)
        iv64 = Intervention(feat_deltas={k: v.double() for k, v in iv.feat_deltas.items()})
        truth64 = whatif_forward(model64, gs64, iv64)

        def maxerr(a, b):
            return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)

        noise = maxerr(scratch32, truth64)          # float32's own error
        delta_err = maxerr(inc, truth64)
        # delta must not add error beyond float32 noise (small slack for the
        # extra ReLU-correction rounding)
        assert delta_err <= 2 * noise + 1e-6, \
            f"delta {delta_err:.2e} exceeds float32 noise {noise:.2e} — possible bug"


@pytest.mark.parametrize("impl", IMPLS)
def test_delta_refuses_topological(env, impl):
    data, schema, gs = env
    model = _model("rgcn", 2)
    cache = build_cache(model, gs.base_view())
    g = torch.Generator().manual_seed(0)
    from src.explore.sampler import sample_topo_iv
    iv = sample_topo_iv(schema, gs, g, TR)
    with pytest.raises(ValueError, match="delta path requires"):
        incremental_forward(model, gs, cache, iv, schema, method="delta", impl=impl)


@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
@pytest.mark.parametrize("L", [1, 2, 4])
@pytest.mark.parametrize("kind", ["feature", "topo", "mixed"])
def test_v2_equals_v1(env, arch, L, kind):
    """v2 changes data access only: same output (within tolerance), the same
    rows recomputed, edges aggregated and sign flips as v1."""
    data, schema, gs = env
    model = _model(arch, L)
    cache = build_cache(model, gs.base_view())
    for iv in _random_ivs(schema, gs, kind, n=3, seed=hash((arch, L, kind, 2)) % 2**31):
        r1 = incremental_forward(model, gs, cache, iv, schema, impl="v1")
        r2 = incremental_forward(model, gs, cache, iv, schema, impl="v2")
        assert_exact(r2.h, r1.h)
        for k in ("method", "nodes_recomputed", "edges_touched", "sign_flips"):
            assert r1.stats[k] == r2.stats[k], k
        assert r1.stats["dirty_frontier"] == r2.stats["dirty_frontier"]


@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
def test_v2_retained_state_serves_v1_consumers(env, arch):
    """The v2 result cache (row overlays + edge overlay) must read back as the
    exact dense state of G^Δ: every layer equals v1's derived cache, the graph
    equals gs.apply(iv), and v1 can resume from it (reuse engine path)."""
    from src.incremental.closure import closure_forward
    from src.incremental.engine import feat_dirty_rows, struct_dirty_rows
    data, schema, gs = env
    model = _model(arch, 2)
    cache = build_cache(model, gs.base_view())
    g = torch.Generator().manual_seed(5)
    from src.explore.sampler import sample_feature_iv, sample_topo_iv
    iv = sample_topo_iv(schema, gs, g, TR, n_add=20, n_del=20)
    r1 = incremental_forward(model, gs, cache, iv, impl="v1")
    r2 = incremental_forward(model, gs, cache, iv, impl="v2")
    assert r2.cache.nbytes() < cache.nbytes()          # overlay, not a full copy
    for l in range(3):
        assert_exact(dict(r2.cache.h[l]), dict(r1.cache.h[l]))
    for l in range(2):
        assert_exact(dict(r2.cache.z[l]), dict(r1.cache.z[l]))
    view = gs.apply(iv)
    for et, ei in view.edge_index_dict.items():
        assert torch.equal(r2.cache.edge_index_dict[et], ei)
    # resume from the v2 state with a further feature residual (v1 consumer)
    f = sample_feature_iv(schema, gs, g, NT, 50)
    from src.data.graph_state import Intervention
    both = Intervention(add_edges=iv.add_edges, del_cols=iv.del_cols,
                        feat_deltas=f.feat_deltas)
    h, _, _ = closure_forward(model, r2.cache, gs.apply(both),
                              feat_dirty_rows(f), struct_dirty_rows(gs, Intervention()))
    assert_exact(h, whatif_forward(model, gs, both))


def test_v2_refuses_non_base_topology(env):
    """v2 reads topology from the base index; a dense cache built on an
    edited graph must be rejected rather than silently mis-read."""
    data, schema, gs = env
    model = _model("rgcn", 2)
    cache = build_cache(model, gs.base_view())
    g = torch.Generator().manual_seed(3)
    from src.explore.sampler import sample_topo_iv
    iv = sample_topo_iv(schema, gs, g, TR)
    edited = incremental_forward(model, gs, cache, iv, impl="v1").cache
    with pytest.raises(ValueError, match="base graph's topology"):
        incremental_forward(model, gs, edited, iv, impl="v2")


def _empty_iv():
    from src.data.graph_state import Intervention
    return Intervention()
