# TorchSim, for an agent working on it

TorchSim simulates MR signals in PyTorch, differentiably. A **signal model** is
the only thing a sequence has to supply; differentiation, device placement,
parameter estimation, model-based reconstruction and sequence design are all
written once against that interface.

`AGENTS.md` and `.github/copilot-instructions.md` are symlinks to this file.
Edit this one. If you find yourself editing either of the others, you have a
broken checkout (Windows without `core.symlinks`), not two documents.

## The shape of the package

| Path | What is in it |
| --- | --- |
| `src/torchsim/sequence/` | The description an acquisition is assembled from — events, operators, builders — and the dispatch that turns one into a kernel launch. |
| `src/torchsim/model/` | What a signal model *is*: the physics, the simulator that orders its events, and the binding that resolves a protocol's structure once and rebinds its values per call. |
| `src/torchsim/_epg_cpu.cpp`, `_perk_cpu.cpp` | The CPU kernels. Every path the GPU has exists here too, and the two agree to float32 round-off. |
| `src/torchsim/sequence/_epg_triton.py`, `estimators/_perk_triton.py` | The GPU kernels. |
| `src/torchsim/simulators/`, `estimators/`, `recon/`, `optim/` | The sequences that ship, and what is built on top of them. |
| `tests/`, `examples/`, `docs/` | Mirrored by subpackage, executed by the gallery, built by Sphinx. |

The shared parameter ABI — read by the Python dispatch, the C++ extension and
the Triton kernels alike — is `src/torchsim/sequence/_parameters.py`. A
parameter added there is added in all three places or in none.

## Commands

```sh
pip install -e ".[dev]"     # the whole toolchain, and it compiles the kernels
pytest tests/               # the suite, with coverage
pytest tests/ -n auto       # across cores
pytest tests/ -m interpreted   # the Triton paths, through Triton's CPU interpreter
pre-commit install          # once per clone, or no hook runs on commit
pre-commit run --all-files  # exactly what CI's Lint job runs
bash scripts/build_docs.sh  # HTML into docs/build/html, examples executed
```

`ruff` is the only style tool: `ruff format` is the formatter and the `I` rules
are the import sorter. There is no black and no isort. Do not add a formatting
argument to a command; `pyproject.toml` is the whole configuration, and
`.pre-commit-config.yaml` pins the ruff the hooks read it with.

A hook that rewrites a file fails the commit and leaves the fix in the working
tree — read it, stage it, commit again. `SKIP=ruff-check git commit` steps
around one hook and `git commit --no-verify` around all of them, but CI runs
the same hooks on every branch, so either only defers the failure.

## Traps worth knowing before you spend an hour on one

**A failed extension build leaves the previous `.so` importable.** The suite
then runs green against a kernel that no longer matches the source you are
reading. Read the exit status of the install, not the last lines of its output:

```sh
pip install -e ".[dev]" ; echo "exit: $?"
python -c "import torchsim._epg_cpu as k; print(k.__file__)"
```

**Kernel compiles dominate a cold GPU run**, not the arithmetic. A suite that
takes minutes on a card is mostly Triton compiling one specialization per
feature combination it meets. Run the whole suite at natural boundaries, not
after every edit.

**Triton's cache keys on the source of the kernel file, not on its meaning.**
Reformatting `_epg_triton.py`, or deleting a dead local from it, invalidates
every specialization exactly as a new `tl.constexpr` would: the next full run
on a card goes from minutes to the better part of an hour, almost all of it one
MLIR pass. Know that before letting a formatter touch that file, and say which
number you are quoting.

**The `interpreted` marker is deselected by default.** Those tests run a Triton
kernel through Triton's CPU interpreter — no GPU, no compile, about a minute
each. `TRITON_INTERPRET=1` does the same thing by hand for a script. It is how
the GPU plumbing is verified on a machine with no card.

**`--cov` is on by default** through `addopts`, so a bare `pytest` writes
`coverage.xml`. It is ignored, not tracked.

## Style, and the rule behind it

Formatting is settled by `ruff format`; do not argue with it. What is left is
what a human decides:

- **Write for someone reading the code as it is now.** Never write text whose
  subject is the history of the code — no "used to", no "previously", no "this
  replaces the old X", no naming a bug that has been fixed. Do not justify the
  present shape by contrast with a shape that is gone. Do not restate what the
  code plainly says. This is enforced in review, and it binds prose docs
  exactly as hard as it binds a header comment.
- **A docstring carries what a caller needs to call it**: one line of what it
  does, then Parameters, Returns, Raises, numpydoc. Types belong in the prose,
  where they can be qualified ("array-like, one per echo"); the API pages are
  built with typehints off for that reason, and the annotations stay in the
  source for editors and for mypy.
- A comment earns its place by explaining a non-obvious algorithm, or a choice
  a reader would otherwise undo. Even then, prefer a well-named function, or a
  test whose name states the invariant — those cannot go stale silently. When
  tempted to explain *why not the other way*, **write a test instead**.
- **Units are public at the edges and internal underneath.** A caller writes
  milliseconds, degrees and Hz; a description timestamps in microseconds and
  carries radians. Convert at the boundary, and name the unit in the identifier
  (`duration_s`, `flip_rad`, `t1_ms`) rather than in a comment beside it.

## Changing the physics

A change to what the kernels compute arrives with a test that pins it against
something **outside** TorchSim: a closed form, a published figure, or an
isochromat summation written out in the test itself. The `tests/epg` files are
written that way and each states its invariant in its module docstring. A test
that only compares TorchSim to TorchSim proves the two agree, which was never
in doubt.

Whatever you change in one kernel, change in the other. The C++ and Triton
implementations are held to each other to float32 round-off, and a path that
exists on one side and not the other is a bug in whichever side is missing it.

## Documentation

The pages under `docs/` are **MyST Markdown**, built by Sphinx. One thing stays
reStructuredText because the tooling requires it:

- `docs/_templates/autosummary/*.rst` — `sphinx.ext.autosummary` writes its
  stubs with a hardcoded `.rst` suffix and finds its directives by regex over
  raw source lines, which is also why the API pages hold their `autosummary`
  and `currentmodule` directives inside `{eval-rst}` blocks.

**A gallery header is Markdown behind a one-line shim.** sphinx-gallery pastes
the header *verbatim* into a generated `index.rst`, so a `README.md` handed to
it directly is parsed as reStructuredText: the build succeeds with no warning
and the page silently loses its title, inherits the first subsection's, and
prints MyST labels as literal text. The working arrangement is a `README.rst`
holding nothing but

```rst
.. include:: _gallery_header.md
   :parser: myst_parser.sphinx_
```

with the prose in `_gallery_header.md` beside it. Two settings make that work
and both are load-bearing: `copyfile_regex` in `sphinx_gallery_conf` carries
the `.md` into the output directory so the include resolves, and
`exclude_patterns` keeps Sphinx from also building it as a page of its own,
which would define every label in it twice.

### The gallery's prose

`~/.claude/CLAUDE.md` carries the rules for explanatory prose and they bind
here. Two things are specific to this gallery.

**`examples/01-framework/01-getting-started.py` is the register.** Match it:
a docstring that opens "The scope of this notebook is to…", section titles that
are plain noun phrases naming the operation — *Forward simulation*, *Approaching
steady state*, *Performance tweaking*, *Functional wrapper* — and bodies that
state the thing once, in the second person or the declarative, with no
rhetorical scaffolding around it.

**Every notebook has one scope and stays inside it.** A dictionary match is the
subject of `02-parameter-inference/01`; a notebook that needs one as a baseline
fits it in a hidden cell and gives it a row in a table, not a section. Reverse-
mode derivatives and Cramér–Rao bounds belong to `03-sequence-optimization`, not
to the getting-started page.

**Show the API and hide the plotting.** `sphinx_gallery_start_ignore` is for
figure code and print formatting. A cell a reader would type themselves —
`execution(...)`, `stream=`, `budget_bytes=`, a `description(...)` built by hand
— is visible even when its output is not interesting.

The figures on the explanation pages are simulated at build time by
`docs/explanation_figures.py`, so a figure cannot outlive the behaviour it
shows. To add one, write a function that returns a Matplotlib figure, register
it in the `FIGURES` mapping at the bottom, and reference the file it writes
with a `{figure}` directive. To add a gallery example, drop a script into the
right `examples/` section — the numeric prefix orders it and the module
docstring becomes the page.

An example is executed by the gallery when the interpreter can import
everything it imports, so what a build runs follows from the extras installed
into it, and a page already carrying output executed from the source as it
stands is reused rather than re-run.

`.github/workflows/docs.yml` builds the published pages and GitHub Pages
serves them from `gh-pages`: its HTML job checks every branch with the `doc`
extra, and its Site job executes all thirteen examples with the `examples`
extra and hands the tree to `scripts/publish_docs.py`, which keeps one
directory per version -- `latest` for `main`, its own for each `v*.*.*` tag --
beside the list the version switcher reads.

## Packaging

The build is `scikit-build-core` driving `CMakeLists.txt`; `setup.py` is a shim
for tools that still shell out to it and configures nothing. Both kernels are
plain CPython extensions against the **stable ABI** from 3.10 on: they call no
PyTorch API and link no PyTorch library, which is why one `cp310-abi3` wheel
per platform serves every supported interpreter and why that wheel is a couple
of megabytes rather than the size of libtorch. Keep it that way — a `#include
<torch/...>` in either `.cpp` ends all of that.

Wheels are built by cibuildwheel and published to PyPI by trusted publishing on
a `v*.*.*` tag. `scripts/check_wheel.py` loads each compiled kernel by path,
without importing the package, and is what every built wheel is tested with.
