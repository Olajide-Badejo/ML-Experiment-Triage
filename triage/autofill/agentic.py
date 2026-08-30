"""The agentic form fill demo (5.4): a classifier driving a real browser.

Everything above this module answers a question about a dataset. This one
answers the product question the dataset stands in for: given a page nobody has
seen, how many of its fields get the right value typed into them, and what did
the mistakes cost. The corpus generator already writes standalone HTML pages
with the ground truth on every input (`data-truth`), so the loop is honest end
to end: the fields come out of the page, the values go into the page, and the
score is read back out of the page rather than out of the array the classifier
returned.

**Two transports, one set of logic.** A `PageDriver` is anything that can list
the inputs of a page, type into one of them and say what the inputs now hold.
`ChromePage` is headless Chrome over the DevTools protocol; `StaticPage` is the
same page parsed with `html.parser` and a dictionary standing in for the DOM.
`run_page` is written against the protocol and never learns which one it has,
which is what makes `--no-browser` a real test of the demo rather than a
separate code path that happens to share a name. The one thing the static
driver cannot check is that a live DOM behaves the way the dictionary does, and
that is asserted directly in the browser test: the two extractors have to
report the same fields for the same page.

**The read back is a check, not a formality.** A fill is scored correct only
when the predicted type is the true type AND the value read back out of the
element is the value that was typed. Dispatching `input` and `change` without
the value actually landing is a plausible browser automation failure, and a
demo that scored its own intentions would report a perfect run through it.

**Nothing here leaves the machine.** The pages are `file://` URLs, the profile
is synthetic, and the card fields hold the documented test number
`4242 4242 4242 4242` (Section 10). There is no real site, no real identity and
no network anywhere in this module.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from triage._extras import require
from triage.autofill.evaluate import ScoredEngine
from triage.autofill.generator import FieldRecord
from triage.autofill.policy import RewardModel, ThresholdPolicy
from triage.autofill.taxonomy import FIELD_TYPES, FieldType

#: The extra that ships the browser driver, and what to say it is for.
EXTRA = "agentic"
PURPOSE = "The agentic form fill demo"

#: The two series a run directory logs. Both names carry their direction, so
#: `triage compare` reads them as higher is better without being told, which is
#: the same contract the sweep and the evaluation already rely on.
TASK_TAGS: tuple[str, ...] = ("task/fill_accuracy", "task/reward")

#: The viewport the demo renders at, and the device scale factor it captures
#: at. Scale 1 is D35a: the committed screenshot is displayed at about 900
#: pixels wide and a scale 2 export would be four times the bytes of a figure
#: nobody sees at that resolution.
VIEWPORT_WIDTH = 1000
VIEWPORT_HEIGHT = 900
SCREENSHOT_SCALE = 1

#: Bounds on the height the capture crops to. The floor keeps a nearly empty
#: page from producing a sliver, and the ceiling keeps a page that reports an
#: absurd height (a broken layout, an infinite scroller) from asking the browser
#: for a hundred megapixel image.
MIN_CAPTURE_HEIGHT = 200
MAX_CAPTURE_HEIGHT = 4000

#: How long to wait for a `file://` page to finish loading, and how often to
#: ask. A generated page is a few kilobytes with no external resources, so this
#: is a guard against a browser that never came up rather than a real wait.
READY_TIMEOUT_S = 10.0
READY_POLL_S = 0.02

#: The synthetic identities the demo types, one per locale. Every value is
#: invented, and the ones that resemble real formats use the ranges reserved
#: for exactly this: `example.com` (RFC 2606), the 555-0100 telephone block,
#: and the card number every payment processor documents as a test number
#: (Section 10). Nothing here is anybody's data.
PROFILES: dict[str, dict[FieldType, str]] = {
    "en_US": {
        FieldType.GIVEN_NAME: "Avery",
        FieldType.FAMILY_NAME: "Nakamura",
        FieldType.NAME: "Avery Nakamura",
        FieldType.EMAIL: "avery.nakamura@example.com",
        FieldType.TEL: "+1 555 0100",
        FieldType.ADDRESS_LINE1: "1600 Sample Street",
        FieldType.ADDRESS_LINE2: "Apartment 4B",
        FieldType.ADDRESS_LEVEL2: "Springfield",
        FieldType.ADDRESS_LEVEL1: "IL",
        FieldType.POSTAL_CODE: "62704",
        FieldType.COUNTRY_NAME: "United States",
        FieldType.ORGANIZATION: "Example Labs",
        FieldType.CC_NAME: "AVERY NAKAMURA",
        FieldType.CC_NUMBER: "4242 4242 4242 4242",
        FieldType.CC_EXP_MONTH: "12",
        FieldType.CC_EXP_YEAR: "2030",
        FieldType.CC_CSC: "123",
        FieldType.USERNAME: "avery.nakamura",
        FieldType.UNKNOWN: "",
    },
    "de_DE": {
        FieldType.GIVEN_NAME: "Lena",
        FieldType.FAMILY_NAME: "Bergmann",
        FieldType.NAME: "Lena Bergmann",
        FieldType.EMAIL: "lena.bergmann@example.com",
        FieldType.TEL: "+49 30 5550100",
        FieldType.ADDRESS_LINE1: "Musterstrasse 12",
        FieldType.ADDRESS_LINE2: "Hinterhaus links",
        FieldType.ADDRESS_LEVEL2: "Berlin",
        FieldType.ADDRESS_LEVEL1: "Berlin",
        FieldType.POSTAL_CODE: "10115",
        FieldType.COUNTRY_NAME: "Deutschland",
        FieldType.ORGANIZATION: "Beispiel GmbH",
        FieldType.CC_NAME: "LENA BERGMANN",
        FieldType.CC_NUMBER: "4242 4242 4242 4242",
        FieldType.CC_EXP_MONTH: "12",
        FieldType.CC_EXP_YEAR: "2030",
        FieldType.CC_CSC: "123",
        FieldType.USERNAME: "lena.bergmann",
        FieldType.UNKNOWN: "",
    },
}


def profile_for(locale: str) -> Mapping[FieldType, str]:
    """The synthetic identity for one locale, refusing a locale it has none for."""
    try:
        return PROFILES[locale]
    except KeyError:
        raise ValueError(
            f"no synthetic profile for locale {locale!r}: the demo fills "
            f"{', '.join(sorted(PROFILES))}"
        ) from None


# ------------------------------------------------------------- page extraction


#: The keys every extractor produces per input, whichever transport it used.
#: Plain strings rather than a dataclass because the browser hands them over as
#: JSON and a dictionary is what both sides already have.
ROW_KEYS = (
    "id",
    "name",
    "placeholder",
    "autocomplete",
    "type",
    "label",
    "section",
    "truth",
)


@dataclass(frozen=True)
class PageSource:
    """One page's inputs, in document order, whatever read them.

    `records()` is where the raw attributes become the `FieldRecord` the rest of
    the package already classifies, which is the reason both transports stop at
    a list of dictionaries: the neighbour labels, the taxonomy lookup and the
    identity keys are computed once, here, and cannot differ between a browser
    run and a CI run.
    """

    name: str
    template: str
    locale: str
    rows: tuple[Mapping[str, str], ...]

    def records(self) -> list[FieldRecord]:
        if not self.rows:
            raise ValueError(
                f"page {self.name!r} holds no input elements: there is nothing to fill, and a "
                f"score over no fields would be a number about nothing"
            )
        labels = [str(row["label"]) for row in self.rows]
        records = []
        for position, row in enumerate(self.rows):
            records.append(
                FieldRecord(
                    form_id=f"{self.name}-{position:02d}",
                    form_key=self.name,
                    template=self.template,
                    template_id=f"{self.template}-{self.locale}-page",
                    locale=self.locale,
                    split="page",
                    section=str(row["section"]),
                    position=position,
                    label=labels[position],
                    name=str(row["name"]),
                    element_id=str(row["id"]),
                    placeholder=str(row["placeholder"]),
                    autocomplete=str(row["autocomplete"]),
                    input_type=str(row["type"]),
                    previous_label=labels[position - 1] if position else "",
                    next_label=labels[position + 1] if position + 1 < len(labels) else "",
                    field_type=_truth_of(row, self.name),
                )
            )
        return records


def _truth_of(row: Mapping[str, str], page: str) -> FieldType:
    """The `data-truth` token, refusing a page that carries none.

    A page without the attribute is not a page this demo can score, and reading
    a missing attribute as `unknown` would turn an unscoreable page into a
    plausible looking result.
    """
    token = str(row.get("truth", ""))
    try:
        return FieldType(token)
    except ValueError:
        raise ValueError(
            f"input {row.get('id')!r} on page {page!r} carries data-truth={token!r}, which is "
            f"not a token of this taxonomy: the demo scores against the ground truth the "
            f"generator embedded and cannot score a page that has none"
        ) from None


def _locale_from(lang: str) -> str:
    """`en-US` as the corpus spells it. The page is the only place it is written."""
    return lang.replace("-", "_")


class _FormParser(HTMLParser):
    """The `--no-browser` extractor: the same fields, from the markup.

    `html.parser` and nothing else, per 5.4. The generator writes the label
    immediately before its input with a matching `for`, so a `for` map is enough
    and no heuristic about proximity is needed; a label with no `for` is ignored
    for the same reason the browser would ignore it.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict[str, str]] = []
        self.labels: dict[str, str] = {}
        self.lang = ""
        self.form_id = ""
        self._section = ""
        self._label_for: str | None = None
        self._in_legend = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        if tag == "html":
            self.lang = values.get("lang", "")
        elif tag == "form" and not self.form_id:
            self.form_id = values.get("id", "")
        elif tag == "legend":
            self._in_legend = True
            self._section = ""
        elif tag == "label":
            self._label_for = values.get("for")
            if self._label_for is not None:
                self.labels[self._label_for] = ""
        elif tag == "input":
            self.rows.append(
                {
                    "id": values.get("id", ""),
                    "name": values.get("name", ""),
                    "placeholder": values.get("placeholder", ""),
                    "autocomplete": values.get("autocomplete", ""),
                    "type": values.get("type", ""),
                    "label": "",
                    "section": self._section,
                    "truth": values.get("data-truth", ""),
                }
            )

    def handle_endtag(self, tag: str) -> None:
        if tag == "legend":
            self._in_legend = False
        elif tag == "label":
            self._label_for = None

    def handle_data(self, data: str) -> None:
        if self._in_legend:
            self._section += data.strip()
        elif self._label_for is not None:
            self.labels[self._label_for] += data


def parse_page(html: str, name: str) -> PageSource:
    """One page's fields, read out of the markup with the standard library."""
    parser = _FormParser()
    parser.feed(html)
    parser.close()
    rows = []
    for row in parser.rows:
        merged = dict(row)
        merged["label"] = parser.labels.get(row["id"], "").strip()
        rows.append(merged)
    return PageSource(
        name=name,
        template=parser.form_id or name,
        locale=_locale_from(parser.lang),
        rows=tuple(rows),
    )


# ---------------------------------------------------------------- page drivers


class PageDriver(Protocol):
    """What the fill loop needs from a page, and nothing else."""

    @property
    def source(self) -> PageSource:
        """The fields this page holds, as both transports report them."""

    def rows(self) -> list[Mapping[str, str]]:
        """The raw attribute rows, in document order."""

    def fill(self, element_id: str, value: str) -> None:
        """Type `value` into one element the way a user agent would."""

    def values(self) -> dict[str, str]:
        """What every element holds now, read back from the page."""


class StaticPage:
    """A page parsed with the standard library, with a dictionary for a DOM.

    Every method is the browser's method with the browser removed, which is what
    makes the CI run a test of the demo. `fill` refuses an element that is not
    on the page, because `document.getElementById` returning null is a bug in
    the caller rather than a field that would not take a value.
    """

    def __init__(self, html: str, name: str) -> None:
        self._source = parse_page(html, name)
        self._values = {str(row["id"]): "" for row in self._source.rows}

    @classmethod
    def from_path(cls, path: Path) -> StaticPage:
        return cls(path.read_text(encoding="utf-8"), path.stem)

    @property
    def source(self) -> PageSource:
        return self._source

    def rows(self) -> list[Mapping[str, str]]:
        return list(self._source.rows)

    def fill(self, element_id: str, value: str) -> None:
        if element_id not in self._values:
            raise KeyError(f"no element {element_id!r} on page {self._source.name!r}")
        self._values[element_id] = value

    def values(self) -> dict[str, str]:
        return dict(self._values)


class BrowserError(RuntimeError):
    """The browser refused a command, or the page raised while running one."""


#: The DOM read, as one expression. `getAttribute` rather than the IDL property
#: throughout, so that what the browser reports is the attribute the generator
#: wrote and not the browser's normalisation of it: `el.type` answers `"text"`
#: for a type nobody wrote, and the static parser has no way to agree with that.
EXTRACT_JS = """
(() => {
  const rows = [];
  for (const el of document.querySelectorAll('input')) {
    if (el.getAttribute('type') === 'submit') { continue; }
    const set = el.closest('fieldset');
    const legend = set ? set.querySelector('legend') : null;
    const label = el.labels && el.labels.length ? el.labels[0].textContent : '';
    rows.push({
      id: el.id,
      name: el.getAttribute('name') || '',
      placeholder: el.getAttribute('placeholder') || '',
      autocomplete: el.getAttribute('autocomplete') || '',
      type: el.getAttribute('type') || '',
      label: label.trim(),
      section: legend ? legend.textContent.trim() : '',
      truth: el.getAttribute('data-truth') || ''
    });
  }
  return JSON.stringify({
    lang: document.documentElement.getAttribute('lang') || '',
    form: document.querySelector('form') ? document.querySelector('form').id : '',
    rows: rows
  });
})()
"""

#: The fill, as a user agent performs it: focus, set, tell the page twice, blur.
#: Both events bubble, because a framework listens on the form rather than on
#: the input, and `change` without `input` is what a script that sets `.value`
#: directly produces and is exactly the shape of autofill that page code
#: complains about.
FILL_JS = """
(() => {
  const el = document.getElementById(%(id)s);
  if (!el) { throw new Error('no element ' + %(id)s); }
  el.focus();
  el.value = %(value)s;
  el.dispatchEvent(new Event('input', {bubbles: true}));
  el.dispatchEvent(new Event('change', {bubbles: true}));
  el.blur();
  return el.value;
})()
"""

#: The read back, from the DOM rather than from anything this process remembers.
VALUES_JS = """
(() => {
  const out = {};
  for (const el of document.querySelectorAll('input')) {
    if (el.getAttribute('type') === 'submit') { continue; }
    out[el.id] = el.value;
  }
  return JSON.stringify(out);
})()
"""


#: The height of the rendered content, so the capture can crop to it. Measured
#: from the body's box rather than from `documentElement.scrollHeight`, which
#: never reports less than the viewport and would therefore say "as tall as the
#: window" for every page shorter than the window, which is all of them.
DOCUMENT_HEIGHT_JS = """
(() => {
  const body = document.body;
  if (!body) { return '0'; }
  const box = body.getBoundingClientRect();
  const margin = parseFloat(getComputedStyle(body).marginBottom) || 0;
  return String(Math.ceil(box.bottom + window.scrollY + margin));
})()
"""


def chrome_available() -> bool:
    """True when a Chromium the driver could drive is installed on this machine.

    Used to skip the browser tests rather than fail them, which is the same
    discipline the Ollama tests follow: a machine without the service is not a
    machine with a defect.
    """
    try:
        module = require("choreographer.browsers.chromium", EXTRA, PURPOSE)
    except ImportError:
        return False
    try:
        found = module.Chromium.find_browser(skip_local=False)
    except Exception:  # pragma: no cover - a browser search that raises is a missing browser
        return False
    return bool(found)


class ChromeSession:
    """Headless Chrome, driven synchronously over the DevTools protocol.

    choreographer's usable API is asynchronous, and everything else in this
    package is not. Rather than colour the fill loop `async` for the benefit of
    one transport, this owns an event loop and runs each command to completion
    on it. The demo is a sequence of round trips with nothing to overlap, so
    there is no concurrency being given up here, only syntax.
    """

    def __init__(self, *, headless: bool = True) -> None:
        self._module = require("choreographer", EXTRA, PURPOSE)
        self._loop = asyncio.new_event_loop()
        self._browser: Any = None

    def _run(self, coroutine: Any) -> Any:
        return self._loop.run_until_complete(coroutine)

    def open(self) -> None:
        self._browser = self._module.Browser(headless=True)
        self._run(self._browser.open())

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._run(self._browser.close())
        finally:
            self._browser = None
            self._loop.close()

    def __enter__(self) -> ChromeSession:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def page(self, path: Path) -> ChromePage:
        """Open one generated page over `file://` and wait for it to settle.

        The path is resolved first, because a `file://` URI is absolute by
        definition and `--pages experiments/results/.../pages` on the command
        line is not: without this the demo refuses every relative path with a
        message about URIs rather than opening the page it was handed.
        """
        if self._browser is None:
            raise BrowserError("the browser is not open: use `with ChromeSession() as session`")
        tab = self._run(self._browser.create_tab(path.resolve().as_uri()))
        page = ChromePage(self, tab, path.stem)
        page.prepare()
        return page

    def command(self, tab: Any, method: str, params: Mapping[str, Any] | None = None) -> Any:
        """One DevTools command, with both failure shapes turned into one error."""
        response = self._run(tab.send_command(method, params=dict(params or {})))
        if "error" in response:
            raise BrowserError(f"{method} was refused: {response['error']}")
        return response.get("result", {})

    def evaluate(self, tab: Any, expression: str) -> str:
        """One JavaScript expression, returning the string it evaluated to."""
        result = self.command(
            tab,
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            text = detail.get("exception", {}).get("description") or detail.get("text")
            raise BrowserError(f"the page raised while evaluating: {text}")
        return str(result["result"].get("value", ""))


class ChromePage:
    """One tab holding one generated page."""

    def __init__(self, session: ChromeSession, tab: Any, name: str) -> None:
        self._session = session
        self._tab = tab
        self._name = name
        self._source: PageSource | None = None

    def prepare(self) -> None:
        """Wait for the load, pin the viewport, and read the fields once."""
        self._session.command(
            self._tab,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
                "deviceScaleFactor": SCREENSHOT_SCALE,
                "mobile": False,
            },
        )
        deadline = time.perf_counter() + READY_TIMEOUT_S
        while self._session.evaluate(self._tab, "document.readyState") != "complete":
            if time.perf_counter() > deadline:
                raise BrowserError(
                    f"page {self._name!r} was still loading after {READY_TIMEOUT_S:g} s; a "
                    f"file:// page of a few kilobytes that never finishes is a browser that "
                    f"did not start rather than a slow page"
                )
            time.sleep(READY_POLL_S)
        payload = json.loads(self._session.evaluate(self._tab, EXTRACT_JS))
        self._source = PageSource(
            name=self._name,
            template=str(payload["form"]) or self._name,
            locale=_locale_from(str(payload["lang"])),
            rows=tuple(dict(row) for row in payload["rows"]),
        )

    @property
    def source(self) -> PageSource:
        if self._source is None:  # pragma: no cover - prepare() runs in the constructor path
            raise BrowserError("the page has not been read yet")
        return self._source

    def rows(self) -> list[Mapping[str, str]]:
        return list(self.source.rows)

    def fill(self, element_id: str, value: str) -> None:
        expression = FILL_JS % {"id": json.dumps(element_id), "value": json.dumps(value)}
        self._session.evaluate(self._tab, expression)

    def values(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in json.loads(self._session.evaluate(self._tab, VALUES_JS)).items()
        }

    def screenshot(self, path: Path) -> Path:
        """Capture the page at scale 1 and write it, recompressed if that is possible.

        The viewport is shrunk to the height of the content first. A generated
        form is shorter than the window it renders in, and capturing the window
        would put a few hundred rows of empty white into a committed PNG: bytes
        that are not the picture and that no reader is looking at.
        """
        height = int(float(self._session.evaluate(self._tab, DOCUMENT_HEIGHT_JS) or 0))
        self._session.command(
            self._tab,
            "Emulation.setDeviceMetricsOverride",
            {
                "width": VIEWPORT_WIDTH,
                "height": max(MIN_CAPTURE_HEIGHT, min(height, MAX_CAPTURE_HEIGHT)),
                "deviceScaleFactor": SCREENSHOT_SCALE,
                "mobile": False,
            },
        )
        result = self._session.command(
            self._tab,
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": True, "optimizeForSpeed": False},
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))
        return optimise_png(path)


def optimise_png(path: Path) -> Path:
    """Rewrite a PNG at maximum lossless compression, when that is available.

    D35a in two parts. The capture is at scale 1, which is the part that always
    happens and is most of the saving; re encoding the same pixels at zlib level
    9 takes another fifth off and needs an imaging library this package does not
    depend on. When there is none the file is left exactly as the browser
    encoded it, which is a slightly larger PNG of the same pixels rather than a
    failure.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - depends on what is installed
        return path
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, format="PNG", optimize=True, compress_level=9)
    return path


# -------------------------------------------------------------------- the loop


@dataclass(frozen=True)
class FieldOutcome:
    """What happened to one input, and what it was worth.

    `landed` is separate from `correct` on purpose. A field can be filled with
    the right value and still read back empty, and that is a defect in the
    automation rather than a wrong answer from the classifier; keeping the two
    apart is what lets the summary say which one happened.
    """

    element_id: str
    truth: str
    predicted: str
    confidence: float
    filled: bool
    value: str
    readback: str
    landed: bool
    correct: bool
    reward: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "truth": self.truth,
            "predicted": self.predicted,
            "confidence": round(self.confidence, 4),
            "filled": self.filled,
            "value": self.value,
            "landed": self.landed,
            "correct": self.correct,
            "reward": self.reward,
        }


@dataclass(frozen=True)
class PageResult:
    """One page, filled and scored."""

    page: str
    template: str
    locale: str
    engine: str
    policy: str
    outcomes: tuple[FieldOutcome, ...]
    seconds: float

    @property
    def n_fields(self) -> int:
        return len(self.outcomes)

    @property
    def n_filled(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.filled)

    @property
    def n_landed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.filled and outcome.landed)

    @property
    def n_correct(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.correct)

    @property
    def fill_rate(self) -> float:
        return self.n_filled / self.n_fields if self.n_fields else 0.0

    @property
    def fill_accuracy(self) -> float | None:
        """Correct fills over fills. `None` when nothing was filled.

        `None` rather than zero, because a policy that filled nothing has an
        undefined accuracy rather than a bad one, and reporting zero would make
        never filling look like the worst policy on a table where it is the
        safest.
        """
        return self.n_correct / self.n_filled if self.n_filled else None

    @property
    def accuracy(self) -> float:
        """Classification accuracy over every field, filled or not."""
        correct = sum(1 for outcome in self.outcomes if outcome.predicted == outcome.truth)
        return correct / self.n_fields if self.n_fields else 0.0

    @property
    def total_reward(self) -> float:
        return float(sum(outcome.reward for outcome in self.outcomes))

    @property
    def expected_reward(self) -> float:
        """Reward per FIELD, so pages of different lengths are comparable."""
        return self.total_reward / self.n_fields if self.n_fields else 0.0

    @property
    def correction_cost(self) -> float:
        """Wrong fills per field: what the user has to go back and undo."""
        wrong = sum(1 for outcome in self.outcomes if outcome.filled and not outcome.correct)
        return wrong / self.n_fields if self.n_fields else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "template": self.template,
            "locale": self.locale,
            "engine": self.engine,
            "policy": self.policy,
            "n_fields": self.n_fields,
            "n_filled": self.n_filled,
            "n_landed": self.n_landed,
            "n_correct": self.n_correct,
            "fill_rate": self.fill_rate,
            "fill_accuracy": self.fill_accuracy,
            "accuracy": self.accuracy,
            "expected_reward": self.expected_reward,
            "correction_cost": self.correction_cost,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


Classifier = Callable[[list[FieldRecord]], ScoredEngine]


def run_page(
    driver: PageDriver,
    classify: Classifier,
    policy: ThresholdPolicy,
    reward: RewardModel,
    engine: str,
) -> PageResult:
    """Classify, decide, fill, read back, score. One page, one engine.

    The order matters and is the product's order: every field is classified
    before anything is typed, the policy decides from the classification alone,
    and the score is computed from what the page holds afterwards. Reading the
    page back in one pass at the end rather than after each fill is deliberate
    too: it is what catches a fill that a later fill undid.
    """
    source = driver.source
    records = source.records()
    started = time.perf_counter()
    scored = classify(records)
    if scored.predicted.size != len(records) or scored.confidence.size != len(records):
        raise ValueError(
            f"engine {engine!r} returned {scored.predicted.size} prediction(s) for "
            f"{len(records)} field(s): the loop needs one prediction per field, because every "
            f"decision below is about a field it can name"
        )

    profile = profile_for(source.locale)
    intended: dict[str, str] = {}
    for position, record in enumerate(records):
        predicted = FIELD_TYPES[int(scored.predicted[position])]
        confidence = float(scored.confidence[position])
        value = profile[predicted]
        # A field with nothing to type is not filled, whatever the policy says.
        # `unknown` is the taxonomy's "none of the above", and typing a
        # placeholder into it would be a wrong fill invented by the demo.
        if not value or not policy.decide(predicted, confidence):
            continue
        driver.fill(record.element_id, value)
        intended[record.element_id] = value

    readback = driver.values()
    outcomes = []
    for position, record in enumerate(records):
        predicted = FIELD_TYPES[int(scored.predicted[position])]
        value = intended.get(record.element_id, "")
        filled = record.element_id in intended
        landed = filled and readback.get(record.element_id, "") == value
        correct = bool(filled and landed and predicted is record.field_type)
        if not filled:
            earned = reward.skip
        elif correct:
            earned = reward.correct_fill
        else:
            earned = -reward.penalty(record.field_type)
        outcomes.append(
            FieldOutcome(
                element_id=record.element_id,
                truth=record.field_type.value,
                predicted=predicted.value,
                confidence=float(scored.confidence[position]),
                filled=filled,
                value=value,
                readback=readback.get(record.element_id, ""),
                landed=landed,
                correct=correct,
                reward=float(earned),
            )
        )
    return PageResult(
        page=source.name,
        template=source.template,
        locale=source.locale,
        engine=engine,
        policy=policy.kind,
        outcomes=tuple(outcomes),
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------- the whole run


@dataclass(frozen=True)
class DemoConfig:
    """Everything about one pass over the pages that a caller might change."""

    engine: str = "rules"
    browser: bool = True
    bootstrap: int = 5
    seed: int = 0
    screenshot: Path | None = None
    screenshot_page: str | None = None


@dataclass(frozen=True)
class DemoResult:
    """Every page, scored, plus what the whole pass came to."""

    pages: tuple[PageResult, ...]
    engine: str
    policy: str
    reward: RewardModel
    browser: bool
    seconds: float
    bootstrap: int = 5
    seed: int = 0
    screenshot: Path | None = None

    @property
    def outcomes(self) -> tuple[FieldOutcome, ...]:
        return tuple(outcome for page in self.pages for outcome in page.outcomes)

    @property
    def n_fields(self) -> int:
        return sum(page.n_fields for page in self.pages)

    @property
    def n_filled(self) -> int:
        return sum(page.n_filled for page in self.pages)

    @property
    def n_landed(self) -> int:
        return sum(page.n_landed for page in self.pages)

    @property
    def n_correct(self) -> int:
        return sum(page.n_correct for page in self.pages)

    @property
    def fill_accuracy(self) -> float:
        return self.n_correct / self.n_filled if self.n_filled else 0.0

    @property
    def fill_rate(self) -> float:
        return self.n_filled / self.n_fields if self.n_fields else 0.0

    @property
    def total_reward(self) -> float:
        return float(sum(page.total_reward for page in self.pages))

    @property
    def expected_reward(self) -> float:
        return self.total_reward / self.n_fields if self.n_fields else 0.0

    @property
    def correction_cost(self) -> float:
        wrong = sum(1 for outcome in self.outcomes if outcome.filled and not outcome.correct)
        return wrong / self.n_fields if self.n_fields else 0.0

    @property
    def accuracy(self) -> float:
        correct = sum(1 for outcome in self.outcomes if outcome.predicted == outcome.truth)
        return correct / self.n_fields if self.n_fields else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "policy": self.policy,
            "browser": self.browser,
            "bootstrap": self.bootstrap,
            "seed": self.seed,
            "reward": self.reward.to_dict(),
            "reward_note": self.reward.describe(),
            "screenshot": self.screenshot.name if self.screenshot else None,
            "pages": [page.to_dict() for page in self.pages],
            "totals": {
                "n_pages": len(self.pages),
                "n_fields": self.n_fields,
                "n_filled": self.n_filled,
                "n_landed": self.n_landed,
                "n_correct": self.n_correct,
                "fill_rate": self.fill_rate,
                "fill_accuracy": self.fill_accuracy,
                "accuracy": self.accuracy,
                "expected_reward": self.expected_reward,
                "correction_cost": self.correction_cost,
            },
        }


def run_demo(
    pages: Sequence[Path],
    classify: Classifier,
    policy: ThresholdPolicy,
    reward: RewardModel,
    config: DemoConfig,
) -> DemoResult:
    """Fill every page with one engine, in a browser or without one."""
    paths = list(pages)
    if not paths:
        raise ValueError(
            "no pages to fill: point --pages at the `pages` directory `triage autofill "
            "generate` wrote"
        )
    started = time.perf_counter()
    if not config.browser:
        results = [
            run_page(StaticPage.from_path(path), classify, policy, reward, config.engine)
            for path in paths
        ]
        return DemoResult(
            pages=tuple(results),
            engine=config.engine,
            policy=policy.kind,
            reward=reward,
            browser=False,
            seconds=time.perf_counter() - started,
            bootstrap=config.bootstrap,
            seed=config.seed,
        )

    shot: Path | None = None
    results = []
    with ChromeSession() as session:
        for path in paths:
            page = session.page(path)
            results.append(run_page(page, classify, policy, reward, config.engine))
            if config.screenshot is not None and shot is None and _is_shot_page(path, config):
                shot = page.screenshot(config.screenshot)
    return DemoResult(
        pages=tuple(results),
        engine=config.engine,
        policy=policy.kind,
        reward=reward,
        browser=True,
        seconds=time.perf_counter() - started,
        bootstrap=config.bootstrap,
        seed=config.seed,
        screenshot=shot,
    )


def _is_shot_page(path: Path, config: DemoConfig) -> bool:
    """Whether this is the page the one screenshot is taken of.

    One screenshot, per 5.4: the README needs a picture of the demo, not a
    gallery of six near identical forms.
    """
    return config.screenshot_page is None or path.stem == config.screenshot_page


# ------------------------------------------------------------------- artifacts


def cumulative_curve(outcomes: Sequence[FieldOutcome], order: np.ndarray) -> list[dict[str, float]]:
    """Fill accuracy and reward over growing prefixes of one ordering.

    Every point estimates the same two quantities from more of the same fields,
    so the curve converges rather than improving: it is the shape `triage
    compare` reads, and the final point is the page's score.
    """
    curve = []
    filled = correct = 0
    total = 0.0
    for step, index in enumerate(order, start=1):
        outcome = outcomes[int(index)]
        total += outcome.reward
        if outcome.filled:
            filled += 1
        if outcome.correct:
            correct += 1
        curve.append(
            {
                "step": float(step),
                "task/fill_accuracy": (correct / filled) if filled else 0.0,
                "task/reward": total / step,
            }
        )
    return curve


@dataclass(frozen=True)
class AgenticPaths:
    """Where `write_demo` put everything."""

    root: Path
    runs: list[Path] = field(default_factory=list)
    summary: Path = Path()
    screenshot: Path | None = None


def write_demo(out_dir: Path | str, result: DemoResult) -> AgenticPaths:
    """One run directory per (policy, page, replicate), plus the summary.

    The run directories are the ordinary format, so `triage ingest` reads them
    with no flag and `triage compare` ranks the engines against each other on
    the task metrics rather than on a validation curve. The condition is the
    ENGINE and the replicate index runs over (page, resample) pairs, which is
    what makes the comparison paired: seed k is the same page and the same
    resample of it for every engine, so the difference between two engines is
    not also carrying the difference between two pages.
    """
    root = Path(out_dir)
    runs_root = root / "runs"
    written: list[Path] = []
    for page_index, page in enumerate(result.pages):
        for replicate in range(result.bootstrap):
            rng = np.random.default_rng(
                np.random.SeedSequence(entropy=result.seed, spawn_key=(page_index, replicate))
            )
            order = rng.integers(0, page.n_fields, page.n_fields)
            directory = runs_root / f"{result.engine}_{page.page}_seed{replicate}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "variant": result.engine,
                        "engine": result.engine,
                        "page": page.page,
                        "template": page.template,
                        "locale": page.locale,
                        "policy": page.policy,
                        "browser": result.browser,
                        "n_fields": page.n_fields,
                        "seed": page_index * result.bootstrap + replicate,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (directory / "metrics.jsonl").write_text(
                "".join(
                    json.dumps(
                        {"step": int(point["step"]), **{tag: point[tag] for tag in TASK_TAGS}},
                        sort_keys=True,
                    )
                    + "\n"
                    for point in cumulative_curve(page.outcomes, order)
                ),
                encoding="utf-8",
                newline="\n",
            )
            written.append(directory)

    # Named after the engine, because ranking the engines against each other
    # means running this three times into ONE output directory: the run
    # directories already carry the engine in their names and cannot collide,
    # and a summary called `agentic.json` would quietly be whichever engine ran
    # last.
    summary = root / f"agentic_{result.engine}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return AgenticPaths(root=root, runs=written, summary=summary, screenshot=result.screenshot)


def format_table(result: DemoResult) -> str:
    """The acceptance table of 5.6 criterion 5, as plain text."""
    header = (
        f"{'page':>14}  {'locale':>6}  {'fields':>6}  {'filled':>6}  "
        f"{'fill acc':>8}  {'reward':>8}  {'correction':>10}"
    )
    lines = [header, "-" * len(header)]
    for page in result.pages:
        accuracy = "n/a" if page.fill_accuracy is None else f"{page.fill_accuracy:.4f}"
        lines.append(
            f"{page.template:>14}  {page.locale:>6}  {page.n_fields:>6}  {page.n_filled:>6}  "
            f"{accuracy:>8}  {page.expected_reward:>8.4f}  {page.correction_cost:>10.4f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'all pages':>14}  {'both':>6}  {result.n_fields:>6}  {result.n_filled:>6}  "
        f"{result.fill_accuracy:>8.4f}  {result.expected_reward:>8.4f}  "
        f"{result.correction_cost:>10.4f}"
    )
    return "\n".join(lines)
