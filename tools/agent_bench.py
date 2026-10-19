#!/usr/bin/env python3
"""Agent-quality benchmark: fixed coding tasks run by Codex and opencode against `local-ai serve`.

`client_compat.py` asks whether the clients work at all; this asks how well
the model works as their agent, so a candidate build can be compared with a
baseline. Every task is a small Python repository with a deterministic,
hidden check (tests the agent never sees, protected files, an exact answer),
run through the real client CLI behind the recording proxy of
`client_compat.py`. Each run records:

- `success`: the task's check passed (and the client exited 0 in time);
- `turns`: the agent's model requests (compaction turns are counted apart);
- tool calls and tool-call failures in three categories:
  `wrong_tool` (a name the request did not declare, or one the client
  refused as unavailable), `bad_args` (arguments that are not JSON, do not
  match the declared schema, or that the client rejected; tool-call markup
  leaked into the answer text; a server `invalid tool call` failure) and
  `loop` (repeats of an identical call beyond its second use);
- wall-clock seconds of the client run, and per turn the time to first token
  (TTFT), prompt and cached tokens, output tokens and request duration.

Determinism: task files and prompts are fixed; repositories live at fixed
paths (`--work`, default /tmp/local-ai-agent-bench) so prompts that quote the
working directory are identical between runs; the proxy pins `seed` on every
generation request (`--seeds`); the server runs with a fresh `HOME` per run,
so its disk prompt cache, response store and reasoning key start empty; and
opencode gets `--title`, so no title request runs beside its turns. Tasks run
in a fixed order. What stays nondeterministic: the date in the clients' system
prompts, and numerics that a candidate's kernel changes legitimately alter.

The model is loaded by `serve`, so hold the shared GPU lock for the whole run:

    # Codex 0.163.0: npm install --prefix .scratch/codex @openai/codex@0.163.0
    lockf /tmp/local-ai-gpu.lock python3 tools/agent_bench.py \\
        --codex-bin .scratch/codex/node_modules/.bin/codex --label baseline-b09e22f
    tools/agent_bench.py --list
    tools/agent_bench.py --compare A/results.json B/results.json

Output: `results.json` (every run with its per-turn and per-call records),
`summary.md`, `exchanges.jsonl`, `server.log`, and per run the client log and
the agent's diff under `runs/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import client_compat as cc

ROOT = cc.ROOT
HIDDEN_TEST = "_agent_bench_hidden_test.py"


# ---------------------------------------------------------------- tasks


@dataclass
class Task:
    name: str
    kind: str
    files: dict[str, str]
    prompt: str
    # Fixture files the agent must leave untouched (prefixes; "" protects the
    # whole repository, which must then show no change at all).
    protected: tuple[str, ...] = ()
    # unittest source run against the finished repository; it may import the repository's modules.
    hidden_test: str = ""
    # Whether the repository's own `python3 -m unittest` must pass too.
    visible_tests: bool = False
    # Shell command (run with `sh -c`) that must exit 0 in the finished repository.
    command: str = ""
    # Regexes the final answer must all match.
    answer: tuple[str, ...] = ()
    # Text that must not appear in any repository file afterwards.
    forbidden_text: str = ""


PAGINATION = Task(
    name="fix-pagination",
    kind="debug: trace a failing test to a helper in another module",
    files={
        "shop/__init__.py": "",
        "shop/util.py": (
            '"""Small numeric helpers shared by the shop package."""\n\n\n'
            "def ceil_div(numerator, denominator):\n"
            '    """Integer division rounded up, for a non-negative numerator and a positive denominator."""\n'
            "    return numerator // denominator\n\n\n"
            "def clamp(value, low, high):\n"
            '    """Limit value to the closed range [low, high]."""\n'
            "    return max(low, min(value, high))\n"
        ),
        "shop/pagination.py": (
            '"""Pagination of catalogue listings."""\n\n'
            "from shop.util import ceil_div, clamp\n\n\n"
            "def page_count(total, per_page):\n"
            '    """Pages needed to list `total` items, `per_page` per page; an empty listing still has one page."""\n'
            "    if per_page <= 0:\n"
            '        raise ValueError("per_page must be positive")\n'
            "    return max(1, ceil_div(total, per_page))\n\n\n"
            "def page_bounds(total, page, per_page):\n"
            '    """(start, stop) slice indices of 1-based `page`; pages past the end show the last page."""\n'
            "    page = clamp(page, 1, page_count(total, per_page))\n"
            "    start = (page - 1) * per_page\n"
            "    return start, min(start + per_page, total)\n"
        ),
        "tests/__init__.py": "",
        "tests/test_pagination.py": (
            "import unittest\n\n"
            "from shop.pagination import page_bounds, page_count\n\n\n"
            "class PageCountTest(unittest.TestCase):\n"
            "    def test_exact_multiple(self):\n"
            "        self.assertEqual(page_count(9, 3), 3)\n\n"
            "    def test_partial_last_page(self):\n"
            "        self.assertEqual(page_count(10, 3), 4)\n\n"
            "    def test_empty(self):\n"
            "        self.assertEqual(page_count(0, 5), 1)\n\n\n"
            "class PageBoundsTest(unittest.TestCase):\n"
            "    def test_last_partial_page(self):\n"
            "        self.assertEqual(page_bounds(10, 4, 3), (9, 10))\n\n"
            "    def test_past_the_end_shows_last_page(self):\n"
            "        self.assertEqual(page_bounds(10, 9, 3), (9, 10))\n\n\n"
            'if __name__ == "__main__":\n    unittest.main()\n'
        ),
    },
    prompt=(
        "The test suite fails. Run `python3 -m unittest -q` to see the failures, find and fix the root "
        "cause in the shop package (do not edit anything under tests/), and run the tests again to confirm "
        "they pass. Finish with a one-sentence summary of the fix."
    ),
    protected=("tests/",),
    visible_tests=True,
    hidden_test=(
        "import unittest\n\nfrom shop.pagination import page_count\nfrom shop.util import ceil_div\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_ceil_div_rounds_up(self):\n"
        "        self.assertEqual([ceil_div(1, 10), ceil_div(11, 5), ceil_div(0, 4), ceil_div(100, 10)], [1, 3, 0, 10])\n\n"
        "    def test_page_count(self):\n"
        "        self.assertEqual([page_count(1, 10), page_count(21, 10), page_count(0, 1)], [1, 3, 1])\n"
    ),
)

SLUGIFY = Task(
    name="implement-slugify",
    kind="feature: implement a function from its docstring specification",
    files={
        "textkit/__init__.py": "from textkit.slug import slugify\n\n__all__ = [\"slugify\"]\n",
        "textkit/slug.py": (
            '"""URL slugs."""\n\n\n'
            "def slugify(text, max_length=None):\n"
            '    """Convert `text` to a URL slug.\n\n'
            "    Rules, applied in order:\n\n"
            "    1. Lowercase the text.\n"
            "    2. Replace every run of characters that are not ASCII letters or ASCII digits\n"
            "       with a single hyphen.\n"
            "    3. Strip hyphens from both ends.\n"
            "    4. If max_length is not None, truncate the slug to at most max_length characters,\n"
            "       then strip any trailing hyphen again.\n"
            '    5. If the result is empty, return "n-a".\n\n'
            '    >>> slugify("Hello, World!")\n'
            "    'hello-world'\n"
            '    """\n'
            "    raise NotImplementedError\n"
        ),
        "README.md": "# textkit\n\nText helpers. `textkit.slugify(text, max_length=None)` makes URL slugs.\n",
    },
    prompt=(
        "Implement `slugify` in textkit/slug.py exactly as its docstring specifies. Keep the signature and use "
        "only the standard library. Check it on a few examples of your own, then reply with a one-sentence summary."
    ),
    hidden_test=(
        "import unittest\n\nfrom textkit import slugify\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_cases(self):\n"
        "        cases = [((\"Hello, World!\",), \"hello-world\"), ((\"  \\u00dcn\\u00efcode Caf\\u00e9  \",), \"n-code-caf\"),\n"
        "                 ((\"abc def ghi\", 8), \"abc-def\"), ((\"!!!\",), \"n-a\"), ((\"\",), \"n-a\"),\n"
        "                 ((\"A--B__C\",), \"a-b-c\"), ((\"Trailing words here\", 9), \"trailing\"),\n"
        "                 ((\"Version 2.0 Released\",), \"version-2-0-released\"), ((\"already-slugged\",), \"already-slugged\")]\n"
        "        for args, expected in cases:\n"
        "            with self.subTest(args=args):\n"
        "                self.assertEqual(slugify(*args), expected)\n"
    ),
)

RENAME = Task(
    name="rename-function",
    kind="refactor: rename a function across modules, tests and docs",
    files={
        "billing/__init__.py": "",
        "billing/calc.py": (
            '"""Invoice arithmetic."""\n\n\n'
            "def calc_total(lines, tax_rate=0.0):\n"
            '    """Sum of quantity * unit_price over `lines`, plus tax at `tax_rate`, rounded to cents."""\n'
            "    subtotal = sum(quantity * price for quantity, price in lines)\n"
            "    return round(subtotal * (1 + tax_rate), 2)\n"
        ),
        "billing/invoice.py": (
            '"""Invoice rendering."""\n\n'
            "from billing.calc import calc_total\n\n\n"
            "def render_invoice(customer, lines, tax_rate=0.0):\n"
            '    """The invoice text for `customer`."""\n'
            "    total = calc_total(lines, tax_rate)\n"
            '    rows = [f"{quantity} x {price:.2f}" for quantity, price in lines]\n'
            '    return "\\n".join([f"Invoice for {customer}", *rows, f"Total: {total:.2f}"])\n'
        ),
        "billing/report.py": (
            '"""Monthly revenue report."""\n\n'
            "from billing import calc\n\n\n"
            "def monthly_revenue(invoices):\n"
            '    """Total revenue of `invoices`, each a (lines, tax_rate) pair."""\n'
            "    return round(sum(calc.calc_total(lines, rate) for lines, rate in invoices), 2)\n"
        ),
        "tests/__init__.py": "",
        "tests/test_billing.py": (
            "import unittest\n\n"
            "from billing.calc import calc_total\n"
            "from billing.invoice import render_invoice\n"
            "from billing.report import monthly_revenue\n\n\n"
            "class BillingTest(unittest.TestCase):\n"
            "    def test_total_with_tax(self):\n"
            "        self.assertEqual(calc_total([(2, 5.0), (1, 2.5)], 0.1), 13.75)\n\n"
            "    def test_invoice_total_line(self):\n"
            '        self.assertTrue(render_invoice("Ada", [(1, 3.0)]).endswith("Total: 3.00"))\n\n'
            "    def test_monthly_revenue(self):\n"
            "        self.assertEqual(monthly_revenue([([(1, 10.0)], 0.0), ([(2, 1.0)], 0.5)]), 13.0)\n\n\n"
            'if __name__ == "__main__":\n    unittest.main()\n'
        ),
        "README.md": "# billing\n\nCall `billing.calc.calc_total(lines, tax_rate)` to total an invoice.\n",
    },
    prompt=(
        "Rename the function `calc_total` to `compute_total` throughout this repository: its definition, every "
        "import and call site, the tests and the README. Do not leave an alias or wrapper under the old name. "
        "Run `python3 -m unittest -q` afterwards to confirm the tests pass."
    ),
    visible_tests=True,
    forbidden_text="calc_total",
    hidden_test=(
        "import unittest\n\nfrom billing import calc\nfrom billing.invoice import render_invoice\n"
        "from billing.report import monthly_revenue\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_renamed(self):\n"
        "        self.assertFalse(hasattr(calc, \"calc_total\"))\n"
        "        self.assertEqual(calc.compute_total([(3, 2.0)], 0.5), 9.0)\n"
        "        self.assertTrue(render_invoice(\"Bo\", [(2, 2.5)], 0.2).endswith(\"Total: 6.00\"))\n"
        "        self.assertEqual(monthly_revenue([([(1, 1.0)], 0.0)]), 1.0)\n"
    ),
)

LOCATE = Task(
    name="locate-answer",
    kind="code search: answer a question that needs three files and a decoy",
    files={
        "README.md": "# orders\n\nOrder service. The current API is orders/api.py; legacy/ is kept for reference and not served.\n",
        "orders/__init__.py": "",
        "orders/limits.py": (
            '"""Per-tier warehouse limits, in units per order."""\n\n'
            "BASE_LIMIT = 250\n\n"
            "WAREHOUSE_LIMITS = {\n"
            '    "default": BASE_LIMIT,\n'
            '    "priority": BASE_LIMIT * 2,\n'
            '    "bulk": BASE_LIMIT * 4,\n'
            "}\n"
        ),
        "orders/errors.py": (
            '"""Order validation errors."""\n\n\n'
            "class OrderError(Exception):\n"
            '    """Base class of order validation errors."""\n\n\n'
            "class UnknownTier(OrderError):\n"
            '    """The order names a tier that does not exist."""\n\n\n'
            "class QuantityExceeded(OrderError):\n"
            '    """The ordered quantity is above the tier\'s warehouse limit."""\n'
        ),
        "orders/validate.py": (
            '"""Order validation."""\n\n'
            "from orders.errors import QuantityExceeded, UnknownTier\n"
            "from orders.limits import WAREHOUSE_LIMITS\n\n\n"
            "def validate(order):\n"
            '    tier = order.get("tier", "default")\n'
            "    if tier not in WAREHOUSE_LIMITS:\n"
            "        raise UnknownTier(tier)\n"
            '    if order["quantity"] > WAREHOUSE_LIMITS[tier]:\n'
            '        raise QuantityExceeded(f"{order[\'quantity\']} > {WAREHOUSE_LIMITS[tier]}")\n'
        ),
        "orders/status.py": (
            '"""HTTP status codes used by the API."""\n\n'
            "CREATED = 201\nBAD_REQUEST = 400\nNOT_FOUND = 404\nCONFLICT = 409\nUNPROCESSABLE = 422\n"
        ),
        "orders/api.py": (
            '"""The order API."""\n\n'
            "from orders import status\n"
            "from orders.errors import OrderError, QuantityExceeded, UnknownTier\n"
            "from orders.validate import validate\n\n"
            "ERROR_STATUS = {UnknownTier: status.NOT_FOUND, QuantityExceeded: status.CONFLICT}\n\n\n"
            "def handle(order):\n"
            "    try:\n"
            "        validate(order)\n"
            "    except OrderError as error:\n"
            "        return ERROR_STATUS.get(type(error), status.BAD_REQUEST), str(error)\n"
            '    return status.CREATED, "created"\n'
        ),
        "legacy/api_v1.py": (
            '"""Version 1 of the API, kept for reference; not served."""\n\n'
            "BULK_LIMIT = 500\n\n\n"
            "def handle(order):\n"
            '    if order["quantity"] > BULK_LIMIT:\n'
            '        return 400, "too many units"\n'
            '    return 200, "ok"\n'
        ),
    },
    prompt=(
        'In this repository, what HTTP status code does the current API (orders/api.py) return when an order for '
        'the "bulk" tier asks for more units than the warehouse allows, and what is the bulk-tier limit? Look it up '
        "in the code and do not modify any files. Reply on one line in exactly this form: status=<code> limit=<units>"
    ),
    protected=("",),
    answer=(r"status\W{0,3}409\b", r"limit\W{0,3}1,?000\b"),
)

LINECOUNT = Task(
    name="add-cli-flag",
    kind="feature: add a command-line option and document it",
    files={
        "linecount.py": (
            "#!/usr/bin/env python3\n"
            '"""Count the lines, words and characters of text files."""\n\n'
            "import argparse\nimport sys\n\n\n"
            "def count(text):\n"
            '    """(lines, words, characters) of `text`."""\n'
            "    lines = text.splitlines()\n"
            "    return len(lines), sum(len(line.split()) for line in lines), len(text)\n\n\n"
            "def main(argv=None):\n"
            "    parser = argparse.ArgumentParser(description=__doc__)\n"
            '    parser.add_argument("files", nargs="+", help="files to count")\n'
            "    args = parser.parse_args(argv)\n"
            "    for name in args.files:\n"
            '        with open(name, encoding="utf-8") as handle:\n'
            "            lines, words, chars = count(handle.read())\n"
            '        print(f"{lines} {words} {chars} {name}")\n'
            "    return 0\n\n\n"
            'if __name__ == "__main__":\n    sys.exit(main())\n'
        ),
        "test_linecount.py": (
            "import io\nimport os\nimport tempfile\nimport unittest\nfrom contextlib import redirect_stdout\n\n"
            "import linecount\n\n\n"
            "class CountTest(unittest.TestCase):\n"
            "    def test_count(self):\n"
            '        self.assertEqual(linecount.count("a b\\nc\\n"), (2, 3, 6))\n\n'
            "    def test_main_prints_counts(self):\n"
            "        with tempfile.TemporaryDirectory() as directory:\n"
            '            path = os.path.join(directory, "x.txt")\n'
            '            with open(path, "w", encoding="utf-8") as handle:\n'
            '                handle.write("x y\\n")\n'
            "            out = io.StringIO()\n"
            "            with redirect_stdout(out):\n"
            "                linecount.main([path])\n"
            '        self.assertEqual(out.getvalue(), f"1 2 4 {path}\\n")\n\n\n'
            'if __name__ == "__main__":\n    unittest.main()\n'
        ),
        "README.md": (
            "# linecount\n\nCounts lines, words and characters, like a small `wc`.\n\n"
            "## Usage\n\n    python3 linecount.py [options] FILE...\n\n"
            "Options:\n\n- `-h`, `--help`: show help and exit.\n"
        ),
    },
    prompt=(
        "Add an `--ignore-blank` option to linecount.py. With it, lines that are empty or contain only "
        "whitespace are not counted as lines; the word and character counts do not change. Without the option "
        "the output must stay exactly as it is now. Document the option in the Options list of README.md's Usage "
        "section, and run the existing tests (`python3 -m unittest -q`) before finishing."
    ),
    protected=("test_linecount.py",),
    visible_tests=True,
    command="grep -q -- '--ignore-blank' README.md",
    hidden_test=(
        "import os\nimport subprocess\nimport sys\nimport tempfile\nimport unittest\n\n\n"
        "def run(*args):\n"
        "    with tempfile.TemporaryDirectory() as directory:\n"
        "        path = os.path.join(directory, \"f.txt\")\n"
        "        with open(path, \"w\", encoding=\"utf-8\") as handle:\n"
        "            handle.write(\"a b\\n\\n   \\nc\\n\")\n"
        "        out = subprocess.run([sys.executable, \"linecount.py\", *args, path], capture_output=True, text=True, check=True).stdout\n"
        "    return out.split()[:3]\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_flag(self):\n"
        "        self.assertEqual(run(\"--ignore-blank\"), [\"2\", \"3\", \"11\"])\n\n"
        "    def test_default_unchanged(self):\n"
        "        self.assertEqual(run(), [\"4\", \"3\", \"11\"])\n"
    ),
)


def _units_module() -> tuple[str, str, str]:
    """A ~600-line conversion module with one mistyped constant, its visible
    tests, and the hidden test. Generated from a fixed table, so identical
    every run."""
    table = [
        ("kilometers", "miles", 0.621371192237334), ("miles", "kilometers", 1.609344),
        ("meters", "feet", 3.280839895013123), ("feet", "meters", 0.3048),
        ("inches", "centimeters", 2.54), ("centimeters", "inches", 0.3937007874015748),
        ("yards", "meters", 0.9144), ("meters", "yards", 1.0936132983377078),
        ("nautical_miles", "kilometers", 1.852), ("kilometers", "nautical_miles", 0.5399568034557235),
        ("kilograms", "pounds", 2.2046226218487757), ("pounds", "kilograms", 0.45359237),
        ("ounces", "grams", 28.349523125), ("grams", "ounces", 0.03527396194958041),
        ("stones", "kilograms", 6.35029318), ("tonnes", "short_tons", 1.1023113109243878),
        ("liters", "us_gallons", 0.26417205235814845), ("us_gallons", "liters", 3.785411784),
        ("milliliters", "us_fluid_ounces", 0.033814022701843), ("us_fluid_ounces", "milliliters", 29.5735295625),
        ("imperial_gallons", "liters", 4.54609), ("cubic_meters", "liters", 1000.0),
        ("hours", "seconds", 3600.0), ("days", "hours", 24.0), ("weeks", "days", 7.0),
        ("minutes", "seconds", 60.0), ("kmh", "mps", 0.2777777777777778), ("mps", "kmh", 3.6),
        ("mph", "kmh", 1.609344), ("knots", "kmh", 1.852),
        ("square_meters", "square_feet", 10.763910416709722), ("square_feet", "square_meters", 0.09290304),
        ("acres", "hectares", 0.40468564224), ("hectares", "acres", 2.471053814671653),
        ("square_miles", "square_kilometers", 2.589988110336), ("bytes", "bits", 8.0),
        ("kibibytes", "bytes", 1024.0), ("mebibytes", "kibibytes", 1024.0), ("gibibytes", "mebibytes", 1024.0),
        ("joules", "calories", 0.23900573613766729), ("calories", "joules", 4.184),
        ("kilowatt_hours", "megajoules", 3.6), ("btus", "joules", 1055.05585262),
        ("pascals", "psi", 0.00014503773773020923), ("psi", "pascals", 6894.757293168361),
        ("bars", "pascals", 100000.0), ("atmospheres", "pascals", 101325.0),
        ("horsepower", "watts", 745.6998715822702), ("watts", "horsepower", 0.0013410220895950279),
        ("degrees", "radians", 0.017453292519943295), ("radians", "degrees", 57.29577951308232),
        ("newtons", "pound_force", 0.22480894309971047), ("pound_force", "newtons", 4.4482216152605),
        ("light_years", "kilometers", 9460730472580.8), ("astronomical_units", "kilometers", 149597870.7),
    ]
    wrong = ("pounds", "kilograms")  # 0.45359237 is mistyped as 0.4535237
    constants, functions = [], []
    for source, target, factor in table:
        constant = f"{source.upper()}_TO_{target.upper()}"
        written = 0.4535237 if (source, target) == wrong else factor
        constants.append(f"{constant} = {written!r}")
        human_source, human_target = source.replace("_", " "), target.replace("_", " ")
        functions.append(
            f"def {source}_to_{target}(value):\n"
            f'    """Convert a quantity in {human_source} to {human_target}.\n\n'
            f"    `value` must be a finite int or float; the result is a float.\n"
            f"    One unit of {human_source} is {factor!r} {human_target}.\n"
            f'    """\n'
            f"    _check(value)\n"
            f"    return value * {constant}\n"
        )
    module = (
        '"""Unit conversions.\n\nEach `<from>_to_<to>` function multiplies by a named factor defined below.\n"""\n\n'
        "import math\n\n" + "\n".join(constants) + "\n\n\n"
        "def _check(value):\n"
        "    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):\n"
        '        raise ValueError(f"not a finite number: {value!r}")\n\n\n'
        + "\n\n".join(functions)
    )
    visible = (
        "import unittest\n\nfrom units import convert\n\n\n"
        "class ConvertTest(unittest.TestCase):\n"
        "    def test_length(self):\n"
        "        self.assertAlmostEqual(convert.miles_to_kilometers(10), 16.09344)\n"
        "        self.assertAlmostEqual(convert.feet_to_meters(100), 30.48)\n\n"
        "    def test_mass(self):\n"
        "        self.assertAlmostEqual(convert.kilograms_to_pounds(1), 2.2046226218487757)\n"
        "        self.assertAlmostEqual(convert.pounds_to_kilograms(10), 4.5359237)\n\n"
        "    def test_rejects_non_numbers(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            convert.hours_to_seconds(\"1\")\n\n\n"
        'if __name__ == "__main__":\n    unittest.main()\n'
    )
    hidden = (
        "import unittest\n\nfrom units import convert\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_factors(self):\n"
        "        self.assertEqual(convert.POUNDS_TO_KILOGRAMS, 0.45359237)\n"
        "        self.assertAlmostEqual(convert.pounds_to_kilograms(3), 1.36077711, places=9)\n"
        "        self.assertAlmostEqual(convert.stones_to_kilograms(2), 12.70058636, places=9)\n"
        "        self.assertEqual(convert.days_to_hours(2), 48.0)\n"
    )
    return module, visible, hidden


_UNITS, _UNITS_VISIBLE, _UNITS_HIDDEN = _units_module()

LARGE_FILE = Task(
    name="fix-large-file",
    kind="debug: find a mistyped constant in a 600-line module",
    files={
        "units/__init__.py": "",
        "units/convert.py": _UNITS,
        "tests/__init__.py": "",
        "tests/test_convert.py": _UNITS_VISIBLE,
    },
    prompt=(
        "One test in this repository fails. Run `python3 -m unittest -q`, find the cause in the units package "
        "and fix it without editing the tests, then rerun the tests. Finish with a one-sentence summary."
    ),
    protected=("tests/",),
    visible_tests=True,
    hidden_test=_UNITS_HIDDEN,
)

CHECK_SCRIPT = Task(
    name="make-check-pass",
    kind="tool errors: read a failing check's output, fix in two rounds",
    files={
        "scripts/check.sh": "#!/bin/sh\nset -e\npython3 tools/lint.py app\npython3 -m unittest -q\n",
        "tools/lint.py": (
            '"""Fail when a public function under the given directory has no docstring."""\n\n'
            "import ast\nimport pathlib\nimport sys\n\n\n"
            "def main(root):\n"
            "    problems = []\n"
            '    for path in sorted(pathlib.Path(root).rglob("*.py")):\n'
            "        tree = ast.parse(path.read_text(encoding=\"utf-8\"))\n"
            "        for node in ast.walk(tree):\n"
            '            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") '
            "and ast.get_docstring(node) is None:\n"
            "                problems.append(f\"{path}:{node.lineno}: function '{node.name}' has no docstring\")\n"
            "    print(\"\\n\".join(problems) or \"lint: ok\")\n"
            "    return 1 if problems else 0\n\n\n"
            'if __name__ == "__main__":\n    sys.exit(main(sys.argv[1]))\n'
        ),
        "app/__init__.py": "",
        "app/geometry.py": (
            '"""Plane geometry."""\n\nimport math\n\n\n'
            "def circle_area(radius):\n"
            '    """Area of a circle of the given radius."""\n'
            "    return math.pi * radius * radius\n\n\n"
            "def rectangle_perimeter(width, height):\n"
            "    return 2 * (width + height)\n\n\n"
            "def _square(value):\n"
            "    return value * value\n"
        ),
        "app/text.py": (
            '"""Text helpers."""\n\n\n'
            "def word_count(text):\n"
            '    """Number of whitespace-separated words in `text`."""\n'
            "    return len(text.split())\n\n\n"
            "def title_case(text):\n"
            '    """Capitalize the first letter of every word, lowercasing the rest."""\n'
            "    return text[:1].upper() + text[1:].lower()\n"
        ),
        "tests/__init__.py": "",
        "tests/test_app.py": (
            "import unittest\n\nfrom app.geometry import circle_area, rectangle_perimeter\n"
            "from app.text import title_case, word_count\n\n\n"
            "class AppTest(unittest.TestCase):\n"
            "    def test_geometry(self):\n"
            "        self.assertAlmostEqual(circle_area(1), 3.141592653589793)\n"
            "        self.assertEqual(rectangle_perimeter(2, 3), 10)\n\n"
            "    def test_text(self):\n"
            '        self.assertEqual(word_count("a b  c"), 3)\n'
            '        self.assertEqual(title_case("hello wORLD"), "Hello World")\n\n\n'
            'if __name__ == "__main__":\n    unittest.main()\n'
        ),
    },
    prompt=(
        "Run `sh scripts/check.sh` and make it pass. Fix problems in the app package only; do not modify "
        "scripts/, tools/ or tests/. Finish with a one-sentence summary."
    ),
    protected=("scripts/", "tools/", "tests/"),
    command="sh scripts/check.sh",
    hidden_test=(
        "import unittest\n\nfrom app.geometry import rectangle_perimeter\nfrom app.text import title_case\n\n\n"
        "class Hidden(unittest.TestCase):\n"
        "    def test_behaviour(self):\n"
        "        self.assertEqual(rectangle_perimeter(1.5, 2), 7.0)\n"
        "        self.assertEqual(title_case(\"the QUICK brown\"), \"The Quick Brown\")\n"
    ),
)

TASKS = [PAGINATION, SLUGIFY, RENAME, LOCATE, LINECOUNT, LARGE_FILE, CHECK_SCRIPT]

# Client name -> (client_compat client, prepare_client variant).
CLIENTS = {
    "codex": ("codex", ""),
    "codex-apply-patch": ("codex", "apply-patch"),
    "opencode": ("opencode", ""),
    "opencode-responses": ("opencode", "responses"),
}


def check_task(task: Task, repo: Path, answer: str) -> tuple[bool, list[str]]:
    """Run the task's deterministic check on a finished repository. Returns
    (passed, failed checks with short evidence)."""
    failures = []
    for name, original in task.files.items():
        if any(name.startswith(prefix) for prefix in task.protected):
            path = repo / name
            if not path.is_file() or path.read_text(errors="replace") != original:
                failures.append(f"protected file changed: {name}")
    if "" in task.protected:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                check=False).stdout
        status = "\n".join(line for line in status.splitlines() if "__pycache__" not in line)
        if status.strip():
            failures.append(f"repository modified: {status.split()[:4]}")
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONDONTWRITEBYTECODE": "1",
           "HOME": str(repo)}
    if task.visible_tests:
        run = subprocess.run([sys.executable, "-m", "unittest", "-q"], cwd=repo, capture_output=True, text=True,
                             timeout=120, env=env, check=False)
        if run.returncode != 0:
            failures.append(f"visible tests: {run.stderr.strip().splitlines()[-1:]}")
    if task.hidden_test:
        (repo / HIDDEN_TEST).write_text(task.hidden_test)
        try:
            run = subprocess.run([sys.executable, "-m", "unittest", "-q", HIDDEN_TEST[:-3]], cwd=repo,
                                 capture_output=True, text=True, timeout=120, env=env, check=False)
        finally:
            (repo / HIDDEN_TEST).unlink()
        if run.returncode != 0:
            failures.append(f"hidden tests: {run.stderr.strip()[-300:]}")
    if task.command:
        run = subprocess.run(["sh", "-c", task.command], cwd=repo, capture_output=True, text=True, timeout=120,
                             env=env, check=False)
        if run.returncode != 0:
            failures.append(f"command `{task.command}` exit {run.returncode}: {(run.stdout + run.stderr)[-300:]}")
    for pattern in task.answer:
        if not re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer lacks /{pattern}/: {answer[:200]!r}")
    if task.forbidden_text:
        hits = [str(p.relative_to(repo)) for p in sorted(repo.rglob("*"))
                if p.is_file() and ".git" not in p.relative_to(repo).parts and "__pycache__" not in p.parts
                and task.forbidden_text in p.read_text(errors="replace")]
        if hits:
            failures.append(f"{task.forbidden_text!r} still in {hits}")
    return not failures, failures


# ---------------------------------------------------------------- analysis

WRONG_TOOL_RESULT = re.compile(r"unsupported call|unavailable tool|unknown tool|no such tool|tool .{0,40}not found",
                               re.IGNORECASE)
BAD_ARGS_RESULT = re.compile(
    r"failed to parse function arguments|invalid arguments|arguments provided to the tool are invalid|"
    r"invalid input for tool|apply_patch verification failed|invalid patch|missing (?:required )?(?:field|argument)",
    re.IGNORECASE)
LEAKED_CALL = re.compile(r"</?tool_call>|<function=|<\|tool|\"arguments\"\s*:\s*[{\"]")
# Polling a running command repeats an identical call by design.
LOOP_EXEMPT = {"write_stdin"}
# Uses of an identical call allowed before each further one counts as a loop
# (running the tests before and after a fix is one legitimate repeat).
LOOP_ALLOWED = 2


def declared_tools(body: dict) -> dict[str, dict]:
    """Name -> `{"kind", "parameters"}` for the tools a request declares
    (Responses and Chat shapes; hosted tools by type)."""
    tools = {}
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name") or tool.get("type")
        tools[name] = {"kind": tool.get("type"), "parameters": function.get("parameters")}
    return tools


def model_calls(exchange: dict) -> tuple[list[dict], str]:
    """The tool calls an exchange's response made, as `{"call_id", "name",
    "arguments" | "input"}`, and its message text."""
    response = cc.response_object(exchange)
    if response is not None:
        calls = []
        for item in response.get("output") or []:
            if item.get("type") == "function_call":
                calls.append({"call_id": item.get("call_id"), "name": item.get("name"),
                              "arguments": item.get("arguments")})
            elif item.get("type") == "custom_tool_call":
                calls.append({"call_id": item.get("call_id"), "name": item.get("name"), "input": item.get("input")})
        return calls, cc.output_text(response)
    body = exchange.get("response_body")
    if isinstance(body, dict):
        message = ((body.get("choices") or [{}])[0]).get("message") or {}
        calls = [{"call_id": c.get("id"), "name": (c.get("function") or {}).get("name"),
                  "arguments": (c.get("function") or {}).get("arguments")} for c in message.get("tool_calls") or []]
        return calls, message.get("content") or ""
    if not isinstance(body, str):
        return [], ""
    by_index: dict[int, dict] = {}
    text = []
    for event in cc.sse_events(body):
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            for part in delta.get("tool_calls") or []:
                call = by_index.setdefault(part.get("index", 0), {"call_id": None, "name": "", "arguments": ""})
                call["call_id"] = part.get("id") or call["call_id"]
                function = part.get("function") or {}
                call["name"] += function.get("name") or ""
                call["arguments"] += function.get("arguments") or ""
    return [by_index[i] for i in sorted(by_index)], "".join(text)


def tool_results(body: object) -> dict[str, str]:
    """call id -> result text, for every tool result a request carries."""
    results = {}
    if not isinstance(body, dict):
        return results
    items = body.get("input") if isinstance(body.get("input"), list) else body.get("messages") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") in ("function_call_output", "custom_tool_call_output"):
            key, output = item.get("call_id"), item.get("output")
        elif item.get("role") == "tool":
            key, output = item.get("tool_call_id"), item.get("content")
        else:
            continue
        if isinstance(output, list):
            output = "".join(p.get("text", "") for p in output if isinstance(p, dict))
        results[key] = output if isinstance(output, str) else json.dumps(output)
    return results


def schema_errors(value: object, schema: object, path: str = "$") -> list[str]:
    """The JSON Schema subset tool parameters use: type, enum, required,
    properties, additionalProperties: false, items."""
    if not isinstance(schema, dict):
        return []
    errors = []
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        checks = {"string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
                  "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                  "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                  "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
                  "null": lambda v: v is None}
        if not any(checks.get(k, lambda v: True)(value) for k in kinds):
            return [f"{path}: expected {'/'.join(kinds)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        errors += [f"{path}: missing required {key!r}" for key in schema.get("required") or [] if key not in value]
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors += schema_errors(item, schema["items"], f"{path}[{index}]")
    return errors


def classify_call(call: dict, tools: dict[str, dict], result: str | None) -> tuple[str | None, str]:
    """(failure category or None, evidence) for one model tool call. Client
    messages are looked for only at the start of the result, where clients
    put them; a command's own output further down does not count."""
    name = call.get("name")
    head = (result or "")[:300]
    if name not in tools:
        return "wrong_tool", f"undeclared tool {name!r}"
    if WRONG_TOOL_RESULT.search(head):
        return "wrong_tool", f"client refused: {head[:160]!r}"
    if "arguments" in call:
        try:
            arguments = json.loads(call["arguments"] or "")
        except ValueError:
            return "bad_args", f"malformed JSON: {str(call['arguments'])[:160]!r}"
        if not isinstance(arguments, dict):
            return "bad_args", f"arguments are not an object: {str(call['arguments'])[:160]!r}"
        errors = schema_errors(arguments, tools[name].get("parameters"))
        if errors:
            return "bad_args", f"schema: {'; '.join(errors)[:200]}"
    if BAD_ARGS_RESULT.search(head):
        return "bad_args", f"client rejected: {head[:160]!r}"
    return None, ""


def call_signature(call: dict) -> str:
    if "input" in call:
        return f"{call.get('name')}:{call.get('input')}"
    try:
        arguments = json.dumps(json.loads(call.get("arguments") or ""), sort_keys=True)
    except ValueError:
        arguments = str(call.get("arguments"))
    return f"{call.get('name')}:{arguments}"


def usage_numbers(exchange: dict) -> tuple[int, int, int]:
    """(prompt tokens, cached prompt tokens, output tokens) of one exchange."""
    response = cc.response_object(exchange)
    if response is not None:
        usage = response.get("usage") or {}
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        return int(usage.get("input_tokens") or 0), int(cached), int(usage.get("output_tokens") or 0)
    body = exchange.get("response_body")
    usage = {}
    if isinstance(body, dict):
        usage = body.get("usage") or {}
    elif isinstance(body, str):
        for event in cc.sse_events(body):
            usage = event.get("usage") or usage
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return int(usage.get("prompt_tokens") or 0), int(cached), int(usage.get("completion_tokens") or 0)


def is_compaction(body: dict) -> bool:
    items = body.get("input") if isinstance(body.get("input"), list) else []
    return bool(items) and isinstance(items[-1], dict) and items[-1].get("type") == "compaction_trigger"


def analyze(exchanges: list[dict]) -> dict:
    """Turns, per-turn timings and tokens, and categorized tool calls of one run."""
    agent = [e for e in exchanges if isinstance(e.get("request_body"), dict) and e["request_body"].get("tools")]
    results: dict[str, str] = {}
    for exchange in exchanges:
        results.update(tool_results(exchange.get("request_body")))
    turns, calls, protocol_errors = [], [], []
    seen: dict[str, int] = {}
    compactions = 0
    for exchange in agent:
        body = exchange["request_body"]
        failure = cc.exchange_failure(exchange)
        if failure:
            protocol_errors.append(failure[:300])
        if is_compaction(body):
            compactions += 1
            continue
        prompt, cached, output = usage_numbers(exchange)
        tools = declared_tools(body)
        made, text = model_calls(exchange)
        turns.append({"seq": exchange.get("seq"), "ttft": exchange.get("first_token_seconds"),
                      "seconds": exchange.get("duration_seconds"), "prompt_tokens": prompt,
                      "cached_tokens": cached, "output_tokens": output, "tool_calls": len(made),
                      "error": failure})
        if failure and re.search(r"tool call", failure, re.IGNORECASE):
            calls.append({"turn": len(turns), "name": None, "category": "bad_args",
                          "evidence": f"server: {failure[:200]}"})
        if LEAKED_CALL.search(text or ""):
            calls.append({"turn": len(turns), "name": None, "category": "bad_args",
                          "evidence": f"tool-call markup in message text: {text[:160]!r}"})
        for call in made:
            category, evidence = classify_call(call, tools, results.get(call.get("call_id")))
            signature = call_signature(call)
            seen[signature] = seen.get(signature, 0) + 1
            if category is None and call.get("name") not in LOOP_EXEMPT and seen[signature] > LOOP_ALLOWED:
                category, evidence = "loop", f"identical call #{seen[signature]}"
            calls.append({"turn": len(turns), "name": call.get("name"), "category": category,
                          "evidence": evidence, "signature": signature[:300]})
    failures = {k: sum(1 for c in calls if c["category"] == k) for k in ("wrong_tool", "bad_args", "loop")}
    ttfts = [t["ttft"] for t in turns if t["ttft"] is not None]
    return {
        "turns": len(turns),
        "compactions": compactions,
        "tool_calls": sum(1 for c in calls if c["name"] is not None),
        "tool_failures": failures,
        "protocol_errors": protocol_errors,
        "ttft_first": ttfts[0] if ttfts else None,
        "ttft_median": statistics.median(ttfts) if ttfts else None,
        "ttft_max": max(ttfts) if ttfts else None,
        "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
        "uncached_prompt_tokens": sum(t["prompt_tokens"] - t["cached_tokens"] for t in turns),
        "output_tokens": sum(t["output_tokens"] for t in turns),
        "max_prompt_tokens": max((t["prompt_tokens"] for t in turns), default=0),
        "per_turn": turns,
        "calls": calls,
    }


# ---------------------------------------------------------------- running


def client_command(ctx: cc.Context, client: str, task: Task, repo: Path, base: Path) -> list[str]:
    if client == "codex":
        return [ctx.codex_bin, "exec", "--json", "--skip-git-repo-check", "-C", str(repo), task.prompt]
    # `--title` skips opencode's title request, which would otherwise run beside the first turn.
    return [*cc.write_jail(base), ctx.opencode_bin, "run", "--format", "json", "--auto", "--title", task.name,
            "--model", f"local-ai/{cc.MODEL_ID}", task.prompt]


def run_one(ctx: cc.Context, out: Path, task: Task, client_name: str, effort: str, seed: int | None) -> dict:
    client, variant = CLIENTS[client_name]
    name = f"{client_name}-{task.name}"
    repo, base, env = cc.prepare_client(ctx, client, name, effort, variant=variant)
    cc.write_files(repo, task.files)
    cc.git_init(repo)
    ctx.recorder.inject = {"seed": seed} if seed is not None else {}
    label = f"{name}-seed{seed}"
    start = ctx.recorder.begin(label)
    log = base / "client.log"
    code, seconds = cc.run_client(ctx, client_command(ctx, client, task, repo, base), repo, env, log)
    exchanges = ctx.recorder.since(start)
    answer = cc.final_answer(exchanges)
    passed, failures = check_task(task, repo, answer)
    metrics = analyze(exchanges)
    keep = out / "runs" / label
    keep.mkdir(parents=True, exist_ok=True)
    shutil.copy(log, keep / "client.log")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=False, capture_output=True)
    diff = subprocess.run(["git", "diff", "--cached", "--stat", "-p"], cwd=repo, check=False,
                          capture_output=True, text=True).stdout
    (keep / "diff.patch").write_text(diff)
    return {
        "task": task.name, "client": client_name, "effort": effort, "seed": seed,
        "success": passed and code == 0, "check_passed": passed, "check_failures": failures,
        "exit_code": code, "timed_out": code == -999, "wall_seconds": round(seconds, 2),
        "answer": answer[:500], **metrics,
    }


def binary_identity(binary: Path) -> dict:
    """The server binary's hash and, when it sits in a checkout's target/, that checkout's commit."""
    identity = {"path": str(binary)}
    if binary.is_file():
        identity["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    for parent in binary.resolve().parents:
        if (parent / ".git").exists():
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=parent, capture_output=True, text=True,
                                    check=False).stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=parent,
                                   capture_output=True, text=True, check=False).stdout.strip()
            identity.update(checkout=str(parent), commit=commit, dirty=bool(dirty))
            break
    return identity


# ---------------------------------------------------------------- reporting


def aggregate(runs: list[dict]) -> dict:
    def median(values):
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 2) if values else None

    failures = {k: sum(r["tool_failures"][k] for r in runs) for k in ("wrong_tool", "bad_args", "loop")}
    return {
        "runs": len(runs),
        "successes": sum(r["success"] for r in runs),
        "success_rate": round(sum(r["success"] for r in runs) / len(runs), 3) if runs else None,
        "turns_mean": round(statistics.fmean(r["turns"] for r in runs), 2) if runs else None,
        "tool_calls": sum(r["tool_calls"] for r in runs),
        "tool_failures": failures,
        "tool_failure_rate": round(sum(failures.values()) / max(1, sum(r["tool_calls"] for r in runs)), 3),
        "protocol_errors": sum(len(r["protocol_errors"]) for r in runs),
        "timeouts": sum(r["timed_out"] for r in runs),
        "wall_seconds_total": round(sum(r["wall_seconds"] for r in runs), 1),
        "wall_seconds_median": median([r["wall_seconds"] for r in runs]),
        "ttft_first_median": median([r["ttft_first"] for r in runs]),
        "ttft_median": median([t["ttft"] for r in runs for t in r["per_turn"]]),
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in runs),
        "uncached_prompt_tokens_total": sum(r["uncached_prompt_tokens"] for r in runs),
    }


def fmt(value, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def summary_markdown(payload: dict) -> str:
    lines = [f"# Agent benchmark: {payload['label']}", "",
             f"server {payload['server']}; clients {payload['versions']}; effort {payload['effort']}; "
             f"seeds {payload['seeds']}", "",
             "| Client | Runs | Success | Turns (mean) | Tool calls | wrong_tool | bad_args | loop | "
             "Protocol errors | Timeouts | Wall s (total) | Wall s (median) | TTFT first (median) | TTFT all (median) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, agg in payload["aggregate"].items():
        f = agg["tool_failures"]
        lines.append(
            f"| {name} | {agg['runs']} | {agg['successes']}/{agg['runs']} | {fmt(agg['turns_mean'])} | "
            f"{agg['tool_calls']} | {f['wrong_tool']} | {f['bad_args']} | {f['loop']} | {agg['protocol_errors']} | "
            f"{agg['timeouts']} | {fmt(agg['wall_seconds_total'], 0)} | {fmt(agg['wall_seconds_median'], 0)} | "
            f"{fmt(agg['ttft_first_median'])} | {fmt(agg['ttft_median'])} |")
    lines += ["", "| Task | Client | Seed | Pass | Turns | Calls | wrong_tool | bad_args | loop | Wall s | "
              "TTFT first | TTFT median | Max prompt tok | Failed checks / tool-call evidence |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in payload["runs"]:
        f = r["tool_failures"]
        evidence = r["check_failures"] + [f"{c['category']}: {c['evidence']}" for c in r["calls"] if c["category"]]
        if r["exit_code"] != 0:
            evidence.append("timed out" if r["timed_out"] else f"client exit {r['exit_code']}")
        detail = "; ".join(evidence).replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(
            f"| {r['task']} | {r['client']} | {r['seed']} | {'PASS' if r['success'] else 'FAIL'} | {r['turns']} | "
            f"{r['tool_calls']} | {f['wrong_tool']} | {f['bad_args']} | {f['loop']} | {fmt(r['wall_seconds'], 0)} | "
            f"{fmt(r['ttft_first'])} | {fmt(r['ttft_median'])} | {r['max_prompt_tokens']} | {detail} |")
    return "\n".join(lines) + "\n"


def write_report(out: Path, payload: dict) -> None:
    clients = sorted({r["client"] for r in payload["runs"]})
    payload["aggregate"] = {c: aggregate([r for r in payload["runs"] if r["client"] == c]) for c in clients}
    payload["aggregate"]["all"] = aggregate(payload["runs"])
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    (out / "summary.md").write_text(summary_markdown(payload))


def compare(baseline_path: Path, candidate_path: Path) -> str:
    """A Markdown comparison of two results.json files, per client and per run."""
    base, cand = (json.loads(Path(p).read_text()) for p in (baseline_path, candidate_path))
    lines = [f"baseline {base['label']} ({base['server'].get('commit', '?')[:9]}) vs "
             f"candidate {cand['label']} ({cand['server'].get('commit', '?')[:9]})", "",
             "| Client | Success | Turns (mean) | wrong_tool | bad_args | loop | Wall s (total) | TTFT first (median) | "
             "TTFT all (median) |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in base["aggregate"]:
        a, b = base["aggregate"][name], cand["aggregate"].get(name)
        if not b:
            continue
        cell = lambda key, digits=1: f"{fmt(a[key], digits)} → {fmt(b[key], digits)}"  # noqa: E731
        fails = " | ".join(f"{a['tool_failures'][k]} → {b['tool_failures'][k]}" for k in ("wrong_tool", "bad_args", "loop"))
        lines.append(f"| {name} | {a['successes']}/{a['runs']} → {b['successes']}/{b['runs']} | {cell('turns_mean')} | "
                     f"{fails} | {cell('wall_seconds_total', 0)} | {cell('ttft_first_median')} | {cell('ttft_median')} |")
    keyed = {(r["task"], r["client"], r["seed"]): r for r in cand["runs"]}
    lines += ["", "| Task | Client | Seed | Pass | Turns | Tool failures | Wall s | TTFT first |",
              "| --- | --- | ---: | --- | --- | --- | --- | --- |"]
    for r in base["runs"]:
        c = keyed.get((r["task"], r["client"], r["seed"]))
        if not c:
            continue
        verdict = lambda run: "PASS" if run["success"] else "FAIL"  # noqa: E731
        lines.append(f"| {r['task']} | {r['client']} | {r['seed']} | {verdict(r)} → {verdict(c)} | "
                     f"{r['turns']} → {c['turns']} | {sum(r['tool_failures'].values())} → {sum(c['tool_failures'].values())} | "
                     f"{fmt(r['wall_seconds'], 0)} → {fmt(c['wall_seconds'], 0)} | {fmt(r['ttft_first'])} → {fmt(c['ttft_first'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server-bin", type=Path, default=ROOT / "target/release/local-ai")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-serve", action="store_true", help="use a server already on --port")
    parser.add_argument("--codex-bin", default=shutil.which("codex"))
    parser.add_argument("--opencode-bin", default=shutil.which("opencode"))
    parser.add_argument("--clients", default="codex,opencode", help=f"comma-separated, from {sorted(CLIENTS)}")
    parser.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    parser.add_argument("--effort", default="none", help="reasoning effort the clients request")
    parser.add_argument("--seeds", default="1", help="comma-separated seeds pinned on every request; each "
                        "seed runs the whole matrix ('default' sends none)")
    parser.add_argument("--timeout", type=float, default=600.0, help="per run, seconds")
    parser.add_argument("--label", default="", help="name of the build under test, kept in the report")
    parser.add_argument("--work", type=Path, default=Path("/tmp/local-ai-agent-bench"),
                        help="fixed directory for the task repositories (wiped per run)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(*args.compare), end="")
        return 0
    tasks = TASKS
    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = [t for t in TASKS if t.name in wanted]
        if len(tasks) != len(wanted):
            parser.error(f"unknown task in {wanted}; known: {[t.name for t in TASKS]}")
    clients = [c for c in args.clients.split(",") if c]
    if unknown := [c for c in clients if c not in CLIENTS]:
        parser.error(f"unknown clients {unknown}")
    if args.list:
        for task in tasks:
            print(f"{task.name:20} {task.kind}")
        return 0
    kinds = sorted({CLIENTS[c][0] for c in clients})
    for kind in kinds:
        if not getattr(args, f"{kind}_bin"):
            parser.error(f"--{kind}-bin is required (not on PATH)")
    seeds = [None if s == "default" else int(s) for s in args.seeds.split(",")]
    out = (args.out or ROOT / ".scratch/artifacts/agent-bench" / time.strftime("run-%Y%m%dT%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    payload = {"label": args.label, "server": binary_identity(args.server_bin),
               "versions": {k: cc.tool_version(getattr(args, f"{k}_bin")) for k in kinds},
               "effort": args.effort, "seeds": args.seeds, "tasks": [t.name for t in tasks], "clients": clients,
               "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": []}
    server = None
    if not args.no_serve:
        # A fresh HOME: empty disk prompt cache, response store and services store.
        home = out / "server-home"
        shutil.rmtree(home, ignore_errors=True)
        home.mkdir()
        env = {k: v for k, v in os.environ.items() if not cc.SECRET_ENV.search(k)}
        env["HOME"] = str(home)
        server = cc.LocalServer(args.server_bin, args.port, out / "server.log",
                                ["--reasoning-key", str(home / "reasoning.key")], env=env)
        server.start()
    recorder = cc.Recorder(args.port, out / "exchanges.jsonl")
    try:
        proxy_port = recorder.start()
        ctx = cc.Context(work, recorder, proxy_port, args.port, server, args.codex_bin, None, args.opencode_bin,
                         args.timeout)
        for seed in seeds:
            for client in clients:
                for task in tasks:
                    print(f"== {client} {task.name} seed={seed}", flush=True)
                    try:
                        run = run_one(ctx, out, task, client, args.effort, seed)
                    except Exception as error:  # noqa: BLE001 -- record and continue with the next run
                        print(f"   harness error: {error!r}", flush=True)
                        continue
                    f = run["tool_failures"]
                    print(f"   {'PASS' if run['success'] else 'FAIL'} turns={run['turns']} calls={run['tool_calls']} "
                          f"wrong_tool={f['wrong_tool']} bad_args={f['bad_args']} loop={f['loop']} "
                          f"wall={run['wall_seconds']:.0f}s ttft_first={fmt(run['ttft_first'])}s "
                          f"{'; '.join(run['check_failures'])[:200]}", flush=True)
                    payload["runs"].append(run)
                    write_report(out, payload)
    finally:
        recorder.stop()
        if server:
            server.stop()
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_report(out, payload)
    print(f"report: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
