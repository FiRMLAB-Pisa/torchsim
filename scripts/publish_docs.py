#!/usr/bin/env python3
"""Place a built documentation tree into the published site, one version deep.

The site is a directory per version -- ``latest`` for the development branch
and the tag for each release -- with a page at the root that sends a reader to
the newest of them, and the list the version switcher reads beside it.

    python scripts/publish_docs.py docs/build/html latest ../pages

The third argument is a checkout of the branch the site is served from. The
version's directory is replaced, the ones beside it are left alone, and the
root is rebuilt from whatever versions are then present.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

#: What a release directory is called: the tag, as it is written.
RELEASE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(.*)$")

#: The development version, which is not a release and sorts before every one.
LATEST = "latest"

#: Written at the root by every deploy, so nothing else there is a version.
GENERATED = ("index.html", "versions.json", ".nojekyll", "binder")

REDIRECT = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>TorchSim documentation</title>
    <meta http-equiv="refresh" content="0; url={target}/">
    <link rel="canonical" href="{target}/">
  </head>
  <body>
    <p>The documentation is at <a href="{target}/">{target}</a>.</p>
  </body>
</html>
"""


def released(name: str) -> tuple[int, int, int, str] | None:
    """The version a release directory names, or ``None`` if it is not one."""
    found = RELEASE.match(name)
    if found is None:
        return None
    major, minor, patch, rest = found.groups()
    return int(major), int(minor), int(patch), rest


def versions(site: Path) -> list[str]:
    """Every version the site holds, newest release first, ``latest`` last."""
    releases = sorted(
        (directory.name for directory in site.iterdir() if directory.is_dir()),
        key=lambda name: released(name) or (),
        reverse=True,
    )
    ordered = [name for name in releases if released(name) is not None]
    if (site / LATEST).is_dir():
        ordered.append(LATEST)
    return ordered


def catalogue(names: list[str], url: str) -> list[dict[str, object]]:
    """The versions as the switcher reads them, the newest release preferred."""
    newest = next((name for name in names if released(name) is not None), None)
    return [
        {
            "name": f"{name} (stable)" if name == newest else name,
            "version": name,
            "url": f"{url}/{name}/",
            **({"preferred": True} if name == newest else {}),
        }
        for name in names
    ]


def prune(site: Path, keep: list[str]) -> None:
    """Leave the site holding its versions and what this script writes."""
    for entry in site.iterdir():
        if entry.name == ".git" or entry.name in GENERATED or entry.name in keep:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def publish(built: Path, version: str, site: Path, url: str) -> list[str]:
    """Put ``built`` in the site as ``version`` and rebuild the root."""
    if not (built / "index.html").is_file():
        raise SystemExit(f"{built} does not look like a built documentation tree")

    target = site / version
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(built, target)

    names = versions(site)
    prune(site, names)

    # Binder builds the branch itself and reads its environment from the root,
    # so the newest version's is what it gets.
    environment = target / "binder"
    if environment.is_dir():
        shutil.rmtree(site / "binder", ignore_errors=True)
        shutil.copytree(environment, site / "binder")

    (site / "versions.json").write_text(
        json.dumps(catalogue(names, url), indent=2) + "\n", encoding="utf-8"
    )
    (site / "index.html").write_text(REDIRECT.format(target=names[0]), encoding="utf-8")
    # Whole directories of the site start with an underscore, which the
    # default Jekyll pipeline would drop.
    (site / ".nojekyll").touch()
    return names


def main(argv: list[str] | None = None) -> int:
    """Publish one built tree into the site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("built", type=Path, help="the tree Sphinx wrote")
    parser.add_argument("version", help="the version it is published as")
    parser.add_argument("site", type=Path, help="a checkout of the site branch")
    parser.add_argument(
        "--url",
        default="https://firmlab-pisa.github.io/torchsim",
        help="where the site is served from",
    )
    arguments = parser.parse_args(argv)
    names = publish(arguments.built, arguments.version, arguments.site, arguments.url)
    print(f"{arguments.version} published; the site holds {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
