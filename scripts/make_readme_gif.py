"""Build the action GIF at the top of the README, out of the real product.

    python scripts/make_readme_gif.py          write assets/images/triage-demo.gif
    nox -s gif                                 the same, through the runner

**Nothing in this file draws anything.** Every frame is a screenshot taken by
headless Chrome of something this repository actually produced. The opening
frames are one generated `de_DE` checkout page being filled a field at a time
by the trained autofill classifier, through the same `ChromePage` driver
`triage autofill agentic` uses; the closing frames are the sections of the HTML
report that `triage report` writes from the committed demo database. A hand
made GIF would be an advertisement; this one is a recording, and it is
regenerable, which is the whole reason it is a script rather than a file
somebody once made.

**The closing frames are whole sections, not a window slid down the page.** A
fixed window at a fixed scroll fraction cuts a chart in half as soon as the
report gains a row, and a picture of a report showing half a figure is worse
than no picture. The rendered document is asked where its own sections are
(`REPORT_REGIONS_JS`), each one is cut out at that box, and anything taller
than the canvas is scaled down rather than cropped. Sliced figures are then
impossible rather than unlikely.

**The fill loop is deliberately not `agentic.run_page`.** That function fills
every field and then reads the page back once, which is the right order for
scoring and the wrong one for a recording: it produces one final state and no
intermediate ones. Here the classification is identical (the same records, the
same model, the same policy) and only the capture is interleaved, so what the
GIF shows is the order the product fills in and the fields it declines to fill.
The two `unknown` fields are skipped on screen for the reason the demo skips
them: the taxonomy's "none of the above" has no value to type.

Pillow is a build time tool here, imported by this script and by nothing that
ships. It is in the development lock (plotly's static export pulls it in) and
is not a dependency of any extra, which is the same footing `optimise_png` in
`triage/autofill/agentic.py` puts it on.

Deterministic apart from Chrome's own rasterisation: one seed for the corpus,
one for the model, a fixed page, a fixed palette and fixed frame durations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import chdir
from pathlib import Path

from PIL import Image

from triage.autofill import agentic
from triage.autofill.evaluate import model_predict
from triage.autofill.features import featurise
from triage.autofill.generator import GeneratorConfig, load_split, write_corpus
from triage.autofill.model import LogisticModel, TrainConfig, train
from triage.autofill.policy import always_fill
from triage.autofill.taxonomy import FIELD_TYPES

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "assets" / "images"
DEFAULT_GIF = IMAGES / "triage-demo.gif"
DEFAULT_WORK = ROOT / "experiments" / "results" / "readme_gif"
DEMO_DATABASE = ROOT / "experiments" / "demo" / "triage.db"
BASELINE = "lr0.0010_bs32"

#: Which of the six generated pages is recorded. The checkout form is the one
#: with all three sections, and `de_DE` is where the model earns its margin over
#: the keyword baseline, which is the arm the documentation reports.
PAGE = "checkout_de_DE"

#: The corpus the classifier is trained on, at the size the published autofill
#: numbers were measured at rather than at the quick demo size.
N_FIELDS = 4000
SEED = 0

#: The canvas every frame is composed onto. 800 is the width GitHub renders a
#: README image at, so the GIF is displayed at its own resolution and neither
#: upscaled nor paying for pixels nobody sees. The height is the tallest whole
#: section of the report, which is the ranked verdict table: sized to the
#: content rather than to a round number, so that section is drawn at the same
#: scale as everything else instead of being shrunk to fit a canvas chosen
#: first. The generated checkout page is shorter and sits on white.
GIF_WIDTH = 800
GIF_HEIGHT = 520

#: How long each kind of frame is held, in milliseconds. The whole animation is
#: about fourteen seconds, which is long enough to read the filled form and
#: short enough that a reader who arrived for the install command is not held.
OPENING_MS = 900
FILL_MS = 420
SETTLED_MS = 1400
REPORT_MS = 1500

#: Colours of the progress strip along the bottom edge, which is the only thing
#: this script draws. The project blue on the project grid grey.
BAR_HEIGHT = 4
BAR_TRACK = (225, 224, 217)
BAR_FILL = (42, 120, 214)

#: A form and a report are flat colour with dark text, so a small palette is
#: lossless in practice and is what keeps the file inside the README budget.
#: One GLOBAL palette, not one per frame: identical palettes are what let
#: Pillow write later frames as differences from the frame before, which on a
#: recording where one input changes per frame is most of the saving.
PALETTE_COLOURS = 64

#: How long to let the report settle before capturing it. Its charts are Plotly
#: figures drawn by script after the document is complete, so `readyState` is
#: not the signal that they are on screen.
REPORT_SETTLE_S = 2.5

#: Which sections of the report the closing frames show, in order, and where
#: each one starts and ends. Measured from the rendered document rather than
#: guessed as a scroll fraction: a fixed window slid down a page cuts a chart in
#: half the moment the page reflows or gains a row, and a chart cut in half is
#: the one thing a picture of a report must not do. The names are read back in
#: the frame log so a bad region is visible rather than silent.
#:
#: Every selector is a structural one the template guarantees: `.lede` and
#: `.tiles` are the summary, `.scroll` wraps each table, `.figure` wraps each
#: Plotly chart, and `footer` holds the methodology block.
#:
#: Each region comes back as four numbers: the top and bottom of the content
#: itself, and then the two lines the crop may not cross, which are the bottom
#: of whatever is above it and the top of whatever is below. Padding a crop by a
#: fixed margin is what puts the first line of the next paragraph into the frame
#: as a sliced band of text, so the padding is allowed to take up slack and
#: nothing more.
REPORT_REGIONS_JS = """
(() => {
  const top = (el) => el.getBoundingClientRect().top + window.scrollY;
  const bottom = (el) => el.getBoundingClientRect().bottom + window.scrollY;
  const span = (...nodes) => {
    const kept = nodes.filter(Boolean);
    if (!kept.length) { return null; }
    const before = kept[0].previousElementSibling;
    const after = kept[kept.length - 1].nextElementSibling;
    return [
      Math.floor(Math.min(...kept.map(top))),
      Math.ceil(Math.max(...kept.map(bottom))),
      Math.ceil(before ? bottom(before) : 0),
      Math.floor(after ? top(after) : bottom(document.body))
    ];
  };
  const heading = (el) => {
    let node = el;
    while (node && !/^H[1-6]$/.test(node.tagName)) { node = node.previousElementSibling; }
    return node;
  };
  const table = document.querySelector('.scroll');
  const figure = document.querySelector('.figure');
  return JSON.stringify({
    summary: span(document.querySelector('.lede'), document.querySelector('.tiles')),
    verdicts: span(heading(table), table),
    curve: span(heading(figure), figure),
    method: span(document.querySelector('footer'))
  });
})()
"""

#: The order the regions are shown in, and the breathing room asked for around
#: each one where the neighbours leave room for it.
REPORT_ORDER = ("summary", "verdicts", "curve", "method")
REGION_PADDING = 18


# ------------------------------------------------------------------- the model


def build_corpus(work: Path, seed: int, n_fields: int) -> Path:
    """Write the synthetic corpus and its HTML pages. Returns the corpus root."""
    corpus = work / "corpus"
    config = GeneratorConfig(n_fields=n_fields, locales=("en_US", "de_DE"))
    write_corpus(corpus, config, seed=seed)
    return corpus


def train_classifier(corpus: Path, seed: int) -> LogisticModel:
    """Train the n gram head the recording fills with, on the generated corpus."""
    train_records = load_split(corpus, "train")
    val_records = load_split(corpus, "val")
    result = train(featurise(train_records), featurise(val_records), TrainConfig(seed=seed))
    final = result.history[-1]
    print(
        f"trained in {result.seconds:.1f} s: val accuracy {final['val/accuracy']:.4f}, "
        f"val macro F1 {final['val/macro_f1']:.4f}"
    )
    return result.model


# ------------------------------------------------------------------ the frames


def fill_frames(
    session: agentic.ChromeSession,
    page_path: Path,
    model: LogisticModel,
    shots: Path,
) -> list[Image.Image]:
    """Open the page, classify it once, then fill and capture one field at a time.

    The policy is `always_fill`, which is what the demo runs without a fitted
    decision policy: every field the classifier has a value for is typed, and
    the ones it reads as `unknown` are not, because there is nothing to type.
    """
    page = session.page(page_path)
    records = page.source.records()
    predicted, confidence, _ = model_predict(model, featurise(records))
    profile = agentic.profile_for(page.source.locale)
    policy = always_fill()

    frames = [_capture(page, shots, 0)]
    filled = 0
    for position, record in enumerate(records):
        field_type = FIELD_TYPES[int(predicted[position])]
        value = profile[field_type]
        if not value or not policy.decide(field_type, float(confidence[position])):
            continue
        page.fill(record.element_id, value)
        filled += 1
        frames.append(_capture(page, shots, filled))
    page.close()
    print(f"filled {filled} of {len(records)} field(s) on {page_path.stem}")
    return frames


def _capture(page: agentic.ChromePage, shots: Path, index: int) -> Image.Image:
    """One full page screenshot, loaded into memory and detached from its file."""
    path = page.screenshot(shots / f"fill-{index:02d}.png")
    with Image.open(path) as opened:
        return opened.convert("RGB")


def write_report(work: Path, database: Path) -> Path:
    """Render the HTML report the closing frames pan over.

    From the committed demo database rather than from a fresh sweep, so the
    report in the GIF is the report in the screenshot beside it and in the PDFs,
    and building the GIF does not depend on a demo having been run first.

    The report prints the database path it was given, verbatim, in its own
    header and footer. Rendered from the repository root with a path relative to
    it, that line reads `experiments/demo/triage.db`; rendered from anywhere
    else it is an absolute path, which on a published GIF is a picture of
    somebody's home directory. So the render is done from the root on purpose.
    """
    from triage.cli import EXIT_OK
    from triage.cli import main as triage

    output = work / "triage_report.html"
    source = database.resolve()
    named = source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
    with chdir(ROOT):
        code = triage(
            ["report", "--database", str(named), "--baseline", BASELINE,
             "--output", str(output), "--quiet"]
        )  # fmt: skip
    if code != EXIT_OK:
        raise SystemExit(f"`triage report` exited {code}; the GIF has no report to pan")
    return output


def report_frames(session: agentic.ChromeSession, report: Path, shots: Path) -> list[Image.Image]:
    """One frame per whole section of the rendered report, in reading order.

    The page is captured once, at its full height, and the sections are cut out
    of that one image at the boxes the document itself reports. The capture
    happens BEFORE the boxes are read, because `ChromePage.screenshot` resizes
    the viewport to the height of the content and a box measured in one viewport
    is not a box in another.
    """
    page = session.page(report)
    time.sleep(REPORT_SETTLE_S)
    path = page.screenshot(shots / "report.png")
    regions = json.loads(page.evaluate(REPORT_REGIONS_JS))
    page.close()

    with Image.open(path) as opened:
        tall = opened.convert("RGB")
    frames = []
    for name in REPORT_ORDER:
        box = regions.get(name)
        if box is None:
            print(f"note: the report has no {name} section; that frame is dropped")
            continue
        content_top, content_bottom, limit_top, limit_bottom = (int(value) for value in box)
        top = min(content_top, max(limit_top, content_top - REGION_PADDING))
        bottom = max(content_bottom, min(limit_bottom, content_bottom + REGION_PADDING))
        top, bottom = max(0, top), min(tall.height, bottom)
        frames.append(tall.crop((0, top, tall.width, bottom)))
        print(f"report frame {name}: {tall.width}x{bottom - top} at y {top}")
    if not frames:
        raise SystemExit("the rendered report reported none of its own sections")
    return frames


def fit(image: Image.Image) -> Image.Image:
    """One image centred on the canvas, scaled down to fit and never up.

    Scaled rather than cropped, which is what makes the no sliced figures
    property hold by construction: a section taller than the canvas is shown
    smaller, and never shown in halves.

    What is left over is filled with the source's own top left pixel rather than
    with white. The browser form is on white and the report is on an off white
    surface, and a white margin around the report reads as a seam across the
    frame rather than as a margin.
    """
    scale = min(GIF_WIDTH / image.width, GIF_HEIGHT / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS) if scale != 1.0 else image
    canvas = Image.new("RGB", (GIF_WIDTH, GIF_HEIGHT), image.getpixel((0, 0)))
    canvas.paste(resized, ((GIF_WIDTH - size[0]) // 2, (GIF_HEIGHT - size[1]) // 2))
    return canvas


def compose(frames: list[Image.Image]) -> list[Image.Image]:
    """Every frame on the canvas, with the progress strip drawn along the bottom."""
    composed = []
    last = max(1, len(frames) - 1)
    for index, frame in enumerate(frames):
        canvas = fit(frame)
        canvas.paste(BAR_TRACK, (0, GIF_HEIGHT - BAR_HEIGHT, GIF_WIDTH, GIF_HEIGHT))
        width = round(GIF_WIDTH * index / last)
        if width:
            canvas.paste(BAR_FILL, (0, GIF_HEIGHT - BAR_HEIGHT, width, GIF_HEIGHT))
        composed.append(canvas)
    return composed


# --------------------------------------------------------------------- the GIF


def durations(n_fill_frames: int, n_report_frames: int) -> list[int]:
    """How long each frame is held. The last fill is held, so the filled form is read."""
    held = [OPENING_MS] + [FILL_MS] * (n_fill_frames - 1)
    held[-1] = SETTLED_MS
    return held + [REPORT_MS] * n_report_frames


def write_gif(frames: list[Image.Image], held: list[int], path: Path) -> Path:
    """Quantise every frame against one palette and write the animation."""
    palette = _shared_palette(frames)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=indexed[1:],
        duration=held,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


def _shared_palette(frames: list[Image.Image]) -> Image.Image:
    """One palette for the whole animation, sampled from the frames themselves.

    The two halves of the recording do not share a colour scheme: a browser
    form is greys and one blue, and the report adds the chart series. Sampling a
    strip of every frame rather than of the first means the palette is chosen
    over what the animation actually contains.
    """
    strip = Image.new("RGB", (GIF_WIDTH, GIF_HEIGHT * len(frames)), (255, 255, 255))
    for index, frame in enumerate(frames):
        strip.paste(frame, (0, index * GIF_HEIGHT))
    return strip.quantize(colors=PALETTE_COLOURS, method=Image.Quantize.MEDIANCUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_GIF), help="where to write the GIF")
    parser.add_argument("--work", default=str(DEFAULT_WORK), help="scratch directory")
    parser.add_argument("--database", default=str(DEMO_DATABASE), help="the demo database")
    parser.add_argument("--seed", type=int, default=SEED, help="corpus and training seed")
    parser.add_argument("--n-fields", type=int, default=N_FIELDS, help="corpus size")
    parser.add_argument("--keep", action="store_true", help="leave the scratch directory in place")
    args = parser.parse_args()

    reason = agentic.why_no_browser()
    if reason is not None:
        print(f"error: {reason}")
        return 2
    database = Path(args.database)
    if not database.exists():
        print(f"error: no database at {database}; run `nox -s demo` first")
        return 2

    started = time.perf_counter()
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    shots = work / "frames"
    shots.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(work, args.seed, args.n_fields)
    model = train_classifier(corpus, args.seed)
    report = write_report(work, database)

    with agentic.ChromeSession() as session:
        frames = fill_frames(session, corpus / "pages" / f"{PAGE}.html", model, shots)
        closing = report_frames(session, report, shots)

    path = write_gif(
        compose(frames + closing), durations(len(frames), len(closing)), Path(args.out)
    )
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)

    total = sum(durations(len(frames), len(closing))) / 1000
    print(
        f"wrote {path.relative_to(ROOT)}: {len(frames) + len(closing)} frames, "
        f"{GIF_WIDTH}x{GIF_HEIGHT}, {total:.1f} s of animation, "
        f"{path.stat().st_size / 1_048_576:.2f} MB, built in "
        f"{time.perf_counter() - started:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
