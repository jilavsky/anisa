"""MCP tools: path confinement and results (no ``mcp`` package needed), and the server itself."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from anisa import api
from anisa.mcp import tools


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(tools.ROOT_VARIABLE, str(tmp_path))
    return tmp_path


def test_paths_are_confined_to_the_data_root(root, tmp_path_factory):
    (root / "inside.txt").write_text("x")
    assert tools.resolve("inside.txt") == root / "inside.txt"
    assert tools.resolve(root / "inside.txt") == root / "inside.txt"
    outside = tmp_path_factory.mktemp("elsewhere")
    for bad in ("../escape", str(outside), "sub/../../escape"):
        with pytest.raises(PermissionError, match="outside the data root"):
            tools.resolve(bad, must_exist=False)
    with pytest.raises(FileNotFoundError):
        tools.resolve("missing.txt")


def test_a_symlink_out_of_the_root_is_refused(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("target")
    (outside / "secret.anisa.json").write_text("{}")
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        tools.resolve("link/secret.anisa.json")


def test_list_describe_calculate_fit_through_the_tools(root):
    written = tools.write_test_system("prolate", "data", fmt="dat")
    listing = tools.list_files()
    paths = [f["path"] for f in listing["files"]]
    assert (
        "data/prolate/prolate.anisa.json" in paths
        and "data/prolate/prolate_start.anisa.json" in paths
    )
    start = "data/prolate/prolate_start.anisa.json"
    assert tools.describe(start)["fit_parameters"]
    assert (
        0.8
        < tools.calculate("data/prolate/prolate.anisa.json")["statistics"]["reduced_chi_squared"]
        < 1.2
    )
    report = tools.fit(start, results="out/fit")
    assert report["success"] and (root / "out" / "fit.anisa.h5").is_file()
    assert tools.read_results("out/fit.anisa.h5")["fit"]["reduced_chi_squared"] > 0
    assert written["truth"].startswith(str(root))
    json.dumps(report)


def test_write_outside_the_root_is_refused(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere")
    with pytest.raises(PermissionError):
        tools.write_test_system("prolate", str(outside))
    tools.write_test_system("prolate", "data", fmt="dat")
    with pytest.raises(PermissionError):
        tools.fit("data/prolate/prolate_start.anisa.json", results=str(outside / "r"))


def test_requests_are_capped(root):
    tools.write_test_system("prolate", "d", fmt="dat")
    start = "d/prolate/prolate_start.anisa.json"
    with pytest.raises(ValueError, match="limited"):
        tools.fit(start, monte_carlo_runs=10_000)
    with pytest.raises(ValueError, match="limited"):
        tools.anisotropy(start, sweep="omega", fixed=90, q_values=[0.01], n_points=100_000)


def test_every_tool_is_registered_and_documented():
    assert set(tools.TOOLS) >= {"fit", "calculate", "describe", "anisotropy", "list_files"}
    for name, function in tools.TOOLS.items():
        assert function.__doc__, name
    assert set(api.__all__) >= {"fit", "calculate", "describe", "anisotropy"}


# -- the server --------------------------------------------------------------------------------


def test_server_lists_and_calls_tools(root):
    pytest.importorskip("mcp")
    from anisa.mcp.server import mcp

    async def go():
        names = {t.name for t in await mcp.list_tools()}
        content, _ = await mcp.call_tool("anisa_list_test_systems", {})
        return names, content

    names, content = asyncio.run(go())
    assert names == {f"anisa_{n}" for n in tools.TOOLS}
    assert "prolate_oblate" in "".join(c.text for c in content)


def test_stdio_roundtrip(root):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "anisa.mcp.server"],
            env={**os.environ, tools.ROOT_VARIABLE: str(root)},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            made = await session.call_tool(
                "anisa_write_test_system", {"name": "spheres", "fmt": "dat"}
            )
            return [t.name for t in listed.tools], made

    names, made = asyncio.run(asyncio.wait_for(go(), 60))
    assert "anisa_fit" in names and not made.isError
    assert (root / "testdata" / "spheres" / "TRUTH.txt").is_file()
