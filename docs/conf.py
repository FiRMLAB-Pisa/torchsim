"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import ast
import dataclasses
import importlib.util
import os
import re
import shelve
import sys
import tempfile
from pathlib import Path

import sphinx.util.logging
import torch
from sphinx_gallery.sorting import ExplicitOrder
from sphinx_gallery.utils import get_md5sum

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file


def _cores_allowed() -> int:
    """How many cores this machine lets the build use.

    A hosted builder is a slice of a much larger machine, and PyTorch sizes
    its thread pool from the machine: left alone it starts a thread per host
    core, and they spend the build contending for the few the quota allows.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


torch.set_num_threads(min(torch.get_num_threads(), _cores_allowed()))

# -- Project information -----------------------------------------------------

project = "torchsim"
copyright = "2024, TorchSim Contributors"
author = "TorchSim Contributors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx_add_colab_link",
    "sphinx_exec_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# The gallery headers are Markdown pulled into a generated index.rst by an
# include, so they are copied beside it -- and must not also be built as pages
# of their own, or every label in them is defined twice.
exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**/_gallery_header.md",
]

# -- Options for MyST --------------------------------------------------------

#: ``dollarmath`` for the physics, ``colon_fence`` so a directive holding other
#: directives can be written without counting backticks, ``deflist`` for the
#: term-and-description lists the guide pages navigate with, and ``linkify`` so
#: a bare URL is a link, as it is in reStructuredText.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]

#: The footnotes of the explanation pages are already under a References
#: heading, so the rule the transition would draw above them is a second
#: divider where the heading is the first.
myst_footnote_transition = False


# generate autosummary even if no references
autosummary_generate = True
# autosummary_imported_members = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
# Types belong in the docstring, where they are prose a reader can qualify
# ("array-like, one per echo") rather than a signature they have to decode.
# The annotations stay for editors and for mypy.
autodoc_typehints = "none"
# The constructor's arguments are documented in the class docstring, so the
# signature belongs on the class heading directly above them rather than in an
# ``__init__`` entry of its own, which renders with nothing under it.
autodoc_class_signature = "mixed"
# Render a default as the source wrote it, rather than as the repr of the
# object the call produced.
autodoc_preserve_defaults = True

#: ``torch.nn.Module`` is the base of most of the public classes, and its own
#: sixty-odd methods would bury the handful each class actually adds. The
#: autosummary template drops any method whose name is one of these.
autosummary_context = {"inherited_from_torch": sorted(dir(torch.nn.Module))}

napoleon_include_private_with_doc = False
napolon_numpy_docstring = True
napoleon_use_admonition_for_references = True


pygments_style = "sphinx"
highlight_language = "python"

# -- Options for Sphinx Gallery ----------------------------------------------

#: The gallery's sections, in the order a reader should meet them.
GALLERY_SECTIONS = [
    "../examples/01-framework",
    "../examples/02-parameter-inference",
    "../examples/03-sequence-optimization",
    "../examples/04-model-based-imaging",
    "../examples/05-misc",
]


def _importable(module: str) -> bool:
    """Whether this interpreter can import ``module``."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _missing_imports(script: Path) -> list[str]:
    """The top-level modules ``script`` imports and this interpreter lacks."""
    imported: set[str] = set()
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imported.add(node.module.split(".")[0])
    return sorted(name for name in imported if not _importable(name))


#: Every example script, in the order the gallery meets them.
GALLERY_SCRIPTS = [
    script
    for section in GALLERY_SECTIONS
    for script in sorted((Path(__file__).parent / section).glob("[0-9]*.py"))
]

#: Which examples are not executed here, and what each one asked for. An
#: example runs where everything it imports is installed, so an environment
#: holding the ``examples`` extra executes the whole gallery and one holding
#: ``doc`` alone executes what needs nothing but TorchSim. The hosted builder
#: is the second: fetching a subject and segmenting it with a network asks for
#: more memory and more minutes than it is given, so it publishes the output
#: the Docs workflow executed and restored into ``generated/autoexamples``.
UNRUNNABLE = {
    script: missing
    for script in GALLERY_SCRIPTS
    if (missing := _missing_imports(script))
}

#: ``filename_pattern`` is searched in the path of each script, and the file
#: names are unique across the sections.
EXECUTED_PATTERN = (
    "|".join(
        re.escape(script.name) for script in GALLERY_SCRIPTS if script not in UNRUNNABLE
    )
    or r"(?!)"  # nothing to execute, and a pattern that matches nothing
)

sphinx_gallery_conf = {
    "doc_module": "torchsim",
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"torchsim": None},
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["generated/autoexamples"],
    "filename_pattern": EXECUTED_PATTERN,
    "ignore_pattern": r"(__init__|conftest|utils).py",
    "nested_sections": True,
    "subsection_order": ExplicitOrder(GALLERY_SECTIONS),
    "within_subsection_order": "FileNameSortKey",
    # The gallery header is written in Markdown and pulled into the
    # generated index.rst by an include; the file has to travel with it.
    "copyfile_regex": r".*\.md",
    "binder": {
        "org": "firmlab-pisa",
        "repo": "torchsim",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": [
            "./binder/apt.txt",
            "./binder/environment.yml",
        ],
        "notebooks_dir": "examples",
        "use_jupyter_lab": True,
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/FiRMLAB-Pisa/torchsim",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
}

# html_logo = "_static/logos/mri-nufft.png"
# html_favicon = "_static/logos/mri-nufft-icon.png"
html_title = "TorchSim Documentation"


def _skip_undocumented_specials(app, what, name, obj, skip, options):
    """Leave out members that render as a heading with nothing under it.

    A dataclass's synthesized ``__init__`` has no source for
    :confval:`autodoc_preserve_defaults` to read, so its defaults come out as
    reprs -- a whole ``Triggers(excitation=<function Excitation>, ...)`` where
    a reader wants the word ``Triggers``. Every field is documented as an
    attribute, which is where its type is stated anyway. An enum's ``__new__``
    carries no docstring at all.
    """
    if name == "__new__":
        return True  # an enum's, which says nothing a reader wants
    if name != "__init__" or skip:
        return None
    owner = getattr(obj, "__qualname__", "").rsplit(".", 1)[0]
    defined_in = sys.modules.get(getattr(obj, "__module__", ""))
    holder = getattr(defined_in, owner, None)
    return True if holder is not None and dataclasses.is_dataclass(holder) else None


def _hide_ignored_code_from_the_page_only() -> None:
    """Keep the page free of the blocks an example hides, and nothing else.

    sphinx-gallery strips its ignore blocks once, before it writes either the
    page or the notebook, so a downloaded notebook is missing whatever the page
    hides and raises on the first cell that needed it. Stripping them as the
    page is written instead leaves the downloadable script and notebook whole,
    which is what the Binder and Colab links open.

    A cell that is hidden in full renders as nothing rather than as an empty
    ``code-block`` directive. Its *output* -- the figures it drew, what it
    printed -- is emitted separately and is kept either way.
    """
    from sphinx_gallery import gen_rst, py_source_parser

    strip = py_source_parser.remove_ignore_blocks

    def keep(code):
        strip(code)  # for its check that every flag has its partner
        return code

    py_source_parser.remove_ignore_blocks = keep

    original = gen_rst.codestr2rst

    def codestr2rst(code, *args, **kwargs):
        shown = strip(code)
        return original(shown, *args, **kwargs) if shown.strip() else ""

    gen_rst.codestr2rst = codestr2rst

    write_notebook = gen_rst.jupyter_notebook

    def jupyter_notebook(script_blocks, *args, **kwargs):
        """The notebook keeps the code, but not the flags that hid it."""
        return write_notebook(
            [
                block._replace(content=_unflagged(block.content))
                for block in script_blocks
            ],
            *args,
            **kwargs,
        )

    gen_rst.jupyter_notebook = jupyter_notebook


def _unflagged(content: str) -> str:
    """The block without the comment lines that mark a hidden region."""
    return "\n".join(
        line
        for line in content.splitlines()
        if line.strip()
        not in ("# sphinx_gallery_start_ignore", "# sphinx_gallery_end_ignore")
    )


def _draw_explanation_figures(app) -> None:
    """Render the explanation pages' figures with the TorchSim being built.

    They are simulated rather than drawn once and checked in, so a figure on
    those pages cannot outlive the behaviour it shows.
    """
    from explanation_figures import render

    render(os.path.join(app.srcdir, "generated", "figures"))


def _restored(app, script: Path) -> bool:
    """Whether the gallery holds output executed from ``script`` as it stands.

    sphinx-gallery stamps a page it executed with the checksum of the source
    it ran, and reuses the page whenever the two still agree. A page restored
    from the Docs workflow carries that stamp; one rendered from its source
    alone does not.
    """
    stamped = Path(app.srcdir, "generated/autoexamples", script.parent.name)
    stamp = stamped / f"{script.name}.md5"
    return stamp.is_file() and stamp.read_text().strip() == get_md5sum(script, mode="t")


def _say_what_is_not_executed(app) -> None:
    """Name each example this build does not run, and where its output stands."""
    logger = sphinx.util.logging.getLogger(__name__)
    for script, missing in UNRUNNABLE.items():
        if _restored(app, script):
            logger.info(
                "[gallery] %s: publishing the output already executed for it "
                "(no %s here)",
                script.name,
                ", ".join(missing),
            )
        else:
            logger.warning(
                "[gallery] %s is published without its output: no %s, and no "
                "executed copy of it under generated/autoexamples",
                script.name,
                ", ".join(missing),
            )


def _cache_opens(app) -> bool:
    """Whether the cache sphinx-gallery's code-link pass keeps opens here.

    It holds the search indexes it resolves in a :mod:`shelve`, which reaches
    for whichever ``dbm`` backend the interpreter was built with. Some Python
    distributions package one separately, and a cache another backend wrote is
    unreadable whatever this one has.
    """
    cache = Path(app.srcdir, "generated/autoexamples/searchindex")
    try:
        if cache.exists():
            shelve.open(str(cache)).close()
        else:
            with tempfile.TemporaryDirectory() as elsewhere:
                shelve.open(str(Path(elsewhere, "probe"))).close()
    except Exception:
        return False
    return True


def setup(app):
    """Wire in the figure pass, and the passes that only sometimes can run.

    sphinx-gallery's code-link pass is dropped where its cache will not open;
    the links it would add are the only thing lost.
    """
    _hide_ignored_code_from_the_page_only()
    app.connect("builder-inited", _say_what_is_not_executed)
    app.connect("builder-inited", _draw_explanation_figures)
    app.connect("autodoc-skip-member", _skip_undocumented_specials)
    if not _cache_opens(app):
        from sphinx_gallery.docs_resolv import embed_code_links

        for listener in list(app.events.listeners.get("build-finished", [])):
            if listener.handler is embed_code_links:
                app.disconnect(listener.id)
