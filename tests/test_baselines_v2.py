"""The optimised baselines (v2 primitives) must reproduce the reference
implementations: Ripple and NeutronRT-RGCN bitwise (same arithmetic, same
summation order) with the same operation counts; NeutronRT-hGAT (v2 closure)
within the exactness tolerance of its v1 closure, same edge count."""
import pytest
import torch

from src.baselines.harness import BaselineCache, Counters
from src.baselines.methods import NeutronRT, Ripple
from src.eval.exactness import assert_exact
from src.incremental.cache import build_cache
from tests.test_incremental_exactness import _model, _random_ivs, env  # noqa: F401


def _run(m, bc, iv):
    Counters.reset()
    h = m.apply(bc, iv)
    return h, Counters.edges, Counters.verts


@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
@pytest.mark.parametrize("L", [1, 2])
@pytest.mark.parametrize("kind", ["feature", "topo", "mixed"])
def test_optimised_baselines_equal_reference(env, arch, L, kind):
    _, schema, gs = env
    model = _model(arch, L)
    ic, bc = build_cache(model, gs.base_view()), BaselineCache(model, gs)
    for iv in _random_ivs(schema, gs, kind, n=3, seed=hash((arch, L, kind, "b")) % 2**31):
        for cls in (Ripple, NeutronRT):
            h_ref, e_ref, v_ref = _run(cls(model, gs, ic), bc, iv)
            h_opt, e_opt, v_opt = _run(cls(model, gs, ic, impl="optimised"), bc, iv)
            assert e_opt == e_ref
            if cls is NeutronRT and arch == "hgat":
                assert_exact(h_opt, h_ref)
                continue
            assert v_opt == v_ref
            for nt in h_ref:
                assert torch.allclose(h_opt[nt], h_ref[nt], rtol=0, atol=0, equal_nan=True), nt


@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
def test_optimised_baselines_equal_reference_large_topo(env, arch):
    """5% structure update (the pilot's size): Ripple diverges here (non-finite
    rows on hGAT); the optimised path must reproduce those rows too."""
    from src.explore.sampler import sample_topo_iv
    from tests.test_incremental_exactness import TR
    _, schema, gs = env
    model = _model(arch, 2)
    ic, bc = build_cache(model, gs.base_view()), BaselineCache(model, gs)
    n = round(0.05 * gs.edge_index[TR].size(1))
    iv = sample_topo_iv(schema, gs, torch.Generator().manual_seed(42), TR, n_add=n, n_del=n)
    for cls in (Ripple, NeutronRT):
        h_ref, e_ref, _ = _run(cls(model, gs, ic), bc, iv)
        h_opt, e_opt, _ = _run(cls(model, gs, ic, impl="optimised"), bc, iv)
        assert e_opt == e_ref
        for nt in h_ref:
            if cls is NeutronRT and arch == "hgat":
                assert_exact(h_opt[nt], h_ref[nt])
            else:
                assert torch.allclose(h_opt[nt], h_ref[nt], rtol=0, atol=0, equal_nan=True), nt
