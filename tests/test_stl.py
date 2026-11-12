"""Direct checks for the STL core: parsing, metrics, transforms and caching."""

from __future__ import annotations

import time

import pytest

from aerolab.stl import (
    clear_stl_cache,
    inspect_stl,
    silhouette_projected_areas_for_triangles,
    transform_triangles,
    write_binary_stl_triangles,
)


def _cube_triangles(size: float = 2.0):
    half = size / 2.0
    corners = {
        "000": (-half, -half, -half),
        "100": (half, -half, -half),
        "110": (half, half, -half),
        "010": (-half, half, -half),
        "001": (-half, -half, half),
        "101": (half, -half, half),
        "111": (half, half, half),
        "011": (-half, half, half),
    }
    c = corners
    return [
        (c["001"], c["111"], c["101"]),
        (c["001"], c["011"], c["111"]),
        (c["000"], c["100"], c["110"]),
        (c["000"], c["110"], c["010"]),
        (c["000"], c["001"], c["101"]),
        (c["000"], c["101"], c["100"]),
        (c["010"], c["110"], c["111"]),
        (c["010"], c["111"], c["011"]),
        (c["000"], c["010"], c["011"]),
        (c["000"], c["011"], c["001"]),
        (c["100"], c["101"], c["111"]),
        (c["100"], c["111"], c["110"]),
    ]


def test_binary_roundtrip_metrics(tmp_path):
    path = tmp_path / "cube.stl"
    write_binary_stl_triangles(path, _cube_triangles(2.0), header="test cube")
    report = inspect_stl(path)
    assert report.format == "binary"
    assert report.triangle_count == 12
    assert report.unique_vertex_count == 8
    assert report.surface_area == pytest.approx(24.0)
    assert report.volume == pytest.approx(8.0)
    assert report.open_edge_count == 0
    assert report.non_manifold_edge_count == 0
    assert report.degenerate_triangle_count == 0
    assert report.is_cfd_candidate
    assert report.projected_areas.x == pytest.approx(4.0)
    assert report.silhouette_projected_areas.x == pytest.approx(4.0)


def test_ascii_parse_matches_binary(tmp_path):
    triangles = _cube_triangles(2.0)
    ascii_path = tmp_path / "cube_ascii.stl"
    with ascii_path.open("w", encoding="utf-8") as stream:
        stream.write("solid cube\n")
        for triangle in triangles:
            stream.write("  facet normal 0 0 0\n    outer loop\n")
            for vertex in triangle:
                stream.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            stream.write("    endloop\n  endfacet\n")
        stream.write("endsolid cube\n")
    report = inspect_stl(ascii_path)
    assert report.format == "ascii"
    assert report.triangle_count == 12
    assert report.surface_area == pytest.approx(24.0)


def test_transform_scales_and_translates():
    transformed = transform_triangles(
        _cube_triangles(2.0),
        scale=0.5,
        translation=(1.0, 2.0, 3.0),
    )
    xs = [vertex[0] for triangle in transformed for vertex in triangle]
    zs = [vertex[2] for triangle in transformed for vertex in triangle]
    assert min(xs) == pytest.approx(0.5)
    assert max(xs) == pytest.approx(1.5)
    assert min(zs) == pytest.approx(2.5)
    assert max(zs) == pytest.approx(3.5)


def test_silhouette_union_preserves_concavity():
    # Two 1x1 plates with a 1-unit gap between them: the union must stay
    # 1 + 1 = 2 and not fill the gap like a convex hull would.
    plate = [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    ]
    shifted = [
        tuple((x, y + 2.0, z) for x, y, z in triangle)  # type: ignore[misc]
        for triangle in plate
    ]
    areas = silhouette_projected_areas_for_triangles(plate + shifted, scanline_count=128)
    assert areas.z == pytest.approx(2.0, rel=0.05)


def test_cache_invalidates_on_rewrite(tmp_path):
    path = tmp_path / "cached.stl"
    write_binary_stl_triangles(path, _cube_triangles(2.0))
    first = inspect_stl(path)
    assert first.surface_area == pytest.approx(24.0)
    # Rewrite larger: mtime/size change must drop the cached parse.
    time.sleep(0.01)
    write_binary_stl_triangles(path, _cube_triangles(4.0))
    second = inspect_stl(path)
    assert second.surface_area == pytest.approx(96.0)
    clear_stl_cache()
