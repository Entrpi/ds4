#!/usr/bin/env python3
"""Generate deterministic needle-in-image validation vectors."""

import argparse
import json
import os
import random
import re
import string
import sys

from PIL import Image, ImageDraw, ImageFont


SEED = 20260903
WIDTH = 720
HEIGHT = 540
CATEGORIES = ("bar", "table", "screenshot", "digits", "count", "word")
CATEGORY_OFFSETS = {category: position * 1000 for position, category in enumerate(CATEGORIES)}
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test-vectors",
    "flash-vision-exp",
    "needle",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Vision-Exp needle image set."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="output directory (default: tests/test-vectors/flash-vision-exp/needle)",
    )
    return parser.parse_args()


def resolve_font():
    for path in FONT_CANDIDATES:
        try:
            ImageFont.truetype(path, 32)
        except (OSError, ValueError):
            continue
        return path
    return "PIL default"


def load_font(font_used, size):
    if font_used != "PIL default":
        return ImageFont.truetype(font_used, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def seed_image(category, index):
    random.seed(SEED + CATEGORY_OFFSETS[category] + index)


def text_dimensions(draw, text, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top, left, top


def draw_centered(draw, text, center_x, top_y, font, fill):
    width, _, left, bbox_top = text_dimensions(draw, text, font)
    draw.text(
        (int(center_x - width / 2 - left), int(top_y - bbox_top)),
        text,
        font=font,
        fill=fill,
    )


def draw_centered_in_box(draw, text, box, font, fill):
    left, top, right, bottom = box
    width, height, bbox_left, bbox_top = text_dimensions(draw, text, font)
    x = left + (right - left - width) / 2 - bbox_left
    y = top + (bottom - top - height) / 2 - bbox_top
    draw.text((int(x), int(y)), text, font=font, fill=fill)


def save_palette_png(image, path):
    paletted = image.quantize(
        colors=64,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    paletted.info.clear()
    paletted.save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
        bits=8,
    )


def make_item(category, index, question, answer, notes, aliases=None):
    item_id = f"{category}_{index:02d}"
    return {
        "id": item_id,
        "image": f"{item_id}.png",
        "category": category,
        "question": question,
        "answer": str(answer),
        "answer_aliases": list(aliases or []),
        "notes": notes,
    }


def generate_bar(index, font_used):
    seed_image("bar", index)
    image = Image.new("RGB", (WIDTH, HEIGHT), (248, 247, 242))
    draw = ImageDraw.Draw(image)
    title_font = load_font(font_used, 34)
    value_font = load_font(font_used, 30)
    label_font = load_font(font_used, 24)
    axis_font = load_font(font_used, 20)

    label_pools = (
        ("Sales", "Support", "Design", "Research", "Finance", "People"),
        ("North", "South", "East", "West", "Central", "Coastal"),
        ("Quartz", "Maple", "Cedar", "Amber", "Coral", "Silver"),
        ("Alpha", "Beta", "Gamma", "Delta", "Kappa", "Sigma"),
    )
    titles = (
        "Quarterly team totals",
        "Regional response counts",
        "Project completion values",
        "Batch quality readings",
    )
    palette = (
        (38, 115, 181),
        (232, 126, 45),
        (67, 160, 71),
        (180, 74, 160),
        (224, 64, 75),
        (61, 166, 166),
    )
    count = random.randint(4, 6)
    labels = random.sample(label_pools[index - 1], count)
    values = random.sample(range(24, 96), count)
    target_position = random.randrange(count)

    draw_centered(draw, titles[index - 1], WIDTH // 2, 22, title_font, (28, 35, 43))
    chart_left, chart_right = 72, 684
    chart_top, chart_bottom = 108, 420
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(55, 60, 65), width=3)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(55, 60, 65), width=3)
    draw.text((18, 76), "reported value", font=axis_font, fill=(55, 60, 65))
    draw_centered(draw, "category", WIDTH // 2, 492, axis_font, (55, 60, 65))

    slot_width = (chart_right - chart_left) / count
    bar_width = min(62, int(slot_width * 0.58))
    for position, (label, value) in enumerate(zip(labels, values)):
        center_x = chart_left + slot_width * (position + 0.5)
        bar_height = int(270 * value / 100)
        bar_top = chart_bottom - bar_height
        draw.rectangle(
            (
                int(center_x - bar_width / 2),
                bar_top,
                int(center_x + bar_width / 2),
                chart_bottom - 2,
            ),
            fill=palette[position],
            outline=(35, 42, 48),
            width=2,
        )
        draw_centered(draw, str(value), center_x, bar_top - 37, value_font, (20, 24, 28))
        draw_centered(draw, label, center_x, 438, label_font, (30, 36, 42))

    target_label = labels[target_position]
    target_value = str(values[target_position])
    item = make_item(
        "bar",
        index,
        f"What is the value of the {target_label} bar?",
        target_value,
        f"A {count}-bar chart with printed values; query targets {target_label}.",
    )
    return image, item


TABLE_SPECS = (
    {
        "title": "Supply catalog",
        "headers": ("Product", "Price", "Stock", "Aisle"),
        "rows": (
            ("Widget A", "12.50", "34", "A2"),
            ("Widget B", "8.75", "19", "C1"),
            ("Widget C", "19.99", "27", "B4"),
            ("Widget D", "15.20", "42", "D3"),
        ),
        "target_row": "Widget C",
        "target_column": "Price",
        "aliases": ("$19.99",),
    },
    {
        "title": "Track practice",
        "headers": ("Runner", "Laps", "Best", "Lane"),
        "rows": (
            ("Mina", "6", "52.8s", "2"),
            ("Theo", "5", "54.1s", "4"),
            ("Ravi", "7", "51.9s", "1"),
            ("Esme", "6", "53.4s", "3"),
            ("Nora", "4", "55.0s", "5"),
        ),
        "target_row": "Mina",
        "target_column": "Best",
        "aliases": ("52.8",),
    },
    {
        "title": "Room sensors",
        "headers": ("Zone", "Temp", "Humidity"),
        "rows": (
            ("North", "21C", "48%"),
            ("West", "24C", "63%"),
            ("South", "22C", "51%"),
        ),
        "target_row": "West",
        "target_column": "Humidity",
        "aliases": ("63",),
    },
    {
        "title": "Workshop queue",
        "headers": ("Item", "Owner", "Status", "Code"),
        "rows": (
            ("Beacon", "Ada", "ready", "K7P4"),
            ("Compass", "Bo", "review", "M8T6"),
            ("Lantern", "Cy", "hold", "R5W3"),
            ("Parcel", "Dee", "ready", "V9N2"),
            ("Signal", "Eli", "draft", "H6Y8"),
        ),
        "target_row": "Beacon",
        "target_column": "Code",
        "aliases": (),
    },
)


def generate_table(index, font_used):
    seed_image("table", index)
    spec = TABLE_SPECS[index - 1]
    rows = list(spec["rows"])
    random.shuffle(rows)
    image = Image.new("RGB", (WIDTH, HEIGHT), (247, 249, 252))
    draw = ImageDraw.Draw(image)
    title_font = load_font(font_used, 34)
    header_font = load_font(font_used, 28)
    cell_font = load_font(font_used, 28)
    draw_centered(draw, spec["title"], WIDTH // 2, 20, title_font, (25, 34, 46))

    grid_left, grid_right, grid_top = 40, 680, 78
    row_height = 68
    column_count = len(spec["headers"])
    boundaries = [
        int(round(grid_left + (grid_right - grid_left) * column / column_count))
        for column in range(column_count + 1)
    ]
    all_rows = [spec["headers"]] + rows
    for row_index, row in enumerate(all_rows):
        top = grid_top + row_index * row_height
        bottom = top + row_height
        background = (43, 91, 138) if row_index == 0 else (
            (236, 242, 248) if row_index % 2 else (250, 252, 254)
        )
        foreground = (255, 255, 255) if row_index == 0 else (25, 31, 38)
        draw.rectangle((grid_left, top, grid_right, bottom), fill=background)
        for column_index, value in enumerate(row):
            cell = (
                boundaries[column_index],
                top,
                boundaries[column_index + 1],
                bottom,
            )
            font = header_font if row_index == 0 else cell_font
            if draw.textlength(value, font=font) > cell[2] - cell[0] - 12:
                raise ValueError(f"table text does not fit: {value}")
            draw_centered_in_box(draw, value, cell, font, foreground)
    grid_bottom = grid_top + len(all_rows) * row_height
    for boundary in boundaries:
        draw.line((boundary, grid_top, boundary, grid_bottom), fill=(55, 66, 78), width=2)
    for row_index in range(len(all_rows) + 1):
        y = grid_top + row_index * row_height
        draw.line((grid_left, y, grid_right, y), fill=(55, 66, 78), width=2)

    row_lookup = {row[0]: row for row in spec["rows"]}
    column_index = spec["headers"].index(spec["target_column"])
    answer = row_lookup[spec["target_row"]][column_index]
    item = make_item(
        "table",
        index,
        f"What is the value in the {spec['target_column']} column for the {spec['target_row']} row?",
        answer,
        f"A {len(rows)}-row table; query targets one named row and column.",
        spec["aliases"],
    )
    return image, item


SCREENSHOT_SPECS = (
    {
        "lines": (
            "08:41 boot: worker service",
            "08:42 info: loading cache",
            "08:43 info: cache ready",
            "08:44 warn: retry count=2",
            "08:45 info: opening queue",
            "08:46 debug: batch size=16",
            "08:47 FATAL: process pid=48213 crashed",
            "08:48 info: writing report",
            "08:49 halt: worker stopped",
        ),
        "question": "What is the PID on the line containing FATAL?",
        "answer": "48213",
        "notes": "A dark worker log with one FATAL line containing a PID.",
    },
    {
        "lines": (
            "# server.conf",
            "host = edge.local",
            "port = 8443",
            "workers = 6",
            "keepalive = true",
            "timeout_ms = 7500",
            "cache_mode = warm",
            "log_level = info",
            "save = enabled",
        ),
        "question": "What value is assigned to timeout_ms?",
        "answer": "7500",
        "notes": "A light configuration editor with a timeout assignment.",
    },
    {
        "lines": (
            "deploy plan: blue",
            "region = south-2",
            "nodes = 8",
            "image = service-a",
            "release_tag = Q7M9K2",
            "health = pending",
            "route = canary",
            "checks = 12",
            "status = staged",
        ),
        "question": "What value is assigned to release_tag?",
        "answer": "Q7M9K2",
        "notes": "A dark deployment summary with a distinctive release tag.",
    },
    {
        "lines": (
            "queue settings",
            "mode: ordered",
            "retry_limit: 7",
            "delay_ms: 250",
            "jitter: false",
            "workers: 4",
            "dead_letter: enabled",
            "metric: queue_depth",
            "state: active",
        ),
        "question": "What value is assigned to retry_limit?",
        "answer": "7",
        "notes": "A light queue settings view with a numeric retry limit.",
    },
)


def generate_screenshot(index, font_used):
    seed_image("screenshot", index)
    spec = SCREENSHOT_SPECS[index - 1]
    dark = index % 2 == 1
    background = (24, 29, 36) if dark else (241, 244, 248)
    foreground = (230, 236, 242) if dark else (30, 37, 45)
    header = (48, 56, 67) if dark else (210, 218, 228)
    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)
    line_font = load_font(font_used, 28)
    draw.rectangle((0, 0, WIDTH, 45), fill=header)
    for dot_index, color in enumerate(((232, 78, 76), (235, 184, 55), (70, 181, 89))):
        x = 23 + dot_index * 28
        draw.ellipse((x - 7, 15, x + 7, 29), fill=color)

    for line_index, line in enumerate(spec["lines"]):
        if draw.textlength(line, font=line_font) > WIDTH - 48:
            raise ValueError(f"screenshot line does not fit: {line}")
        y = 61 + line_index * 49
        draw.text((24, y), line, font=line_font, fill=foreground)

    item = make_item(
        "screenshot",
        index,
        spec["question"],
        spec["answer"],
        spec["notes"],
    )
    return image, item


def generate_digits(index, font_used):
    seed_image("digits", index)
    length = random.randint(6, 10)
    code = "".join(random.choice(CODE_ALPHABET) for _ in range(length))
    backgrounds = (
        (239, 247, 255),
        (255, 247, 229),
        (238, 252, 242),
        (249, 239, 252),
    )
    accents = (
        (31, 89, 151),
        (151, 83, 22),
        (29, 118, 68),
        (112, 53, 132),
    )
    image = Image.new("RGB", (WIDTH, HEIGHT), backgrounds[index - 1])
    draw = ImageDraw.Draw(image)
    label_font = load_font(font_used, 30)
    code_font = load_font(font_used, 78)
    accent = accents[index - 1]
    draw.rounded_rectangle((45, 90, 675, 450), radius=24, outline=accent, width=6)
    draw_centered(draw, "access code", WIDTH // 2, 135, label_font, accent)
    draw_centered(draw, code, WIDTH // 2, 238, code_font, (18, 25, 32))
    item = make_item(
        "digits",
        index,
        "What is the code shown in the image?",
        code,
        f"A centered {length}-character code using only unambiguous glyphs.",
    )
    return image, item


SHAPE_COLORS = {
    "red": (218, 52, 52),
    "blue": (45, 103, 214),
    "green": (42, 163, 79),
    "yellow": (239, 193, 35),
}
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def generate_count(index, font_used):
    seed_image("count", index)
    image = Image.new("RGB", (WIDTH, HEIGHT), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    title_font = load_font(font_used, 32)
    draw_centered(draw, "Count the colored shapes", WIDTH // 2, 24, title_font, (31, 36, 42))

    color_names = tuple(SHAPE_COLORS)
    asked_color = color_names[index - 1]
    total = random.randint(6, 9)
    asked_count = random.randint(2, min(4, total - 1))
    other_colors = [color for color in color_names if color != asked_color]
    assignments = [asked_color] * asked_count + [
        random.choice(other_colors) for _ in range(total - asked_count)
    ]
    random.shuffle(assignments)
    positions = [
        (145, 145),
        (360, 145),
        (575, 145),
        (145, 290),
        (360, 290),
        (575, 290),
        (145, 435),
        (360, 435),
        (575, 435),
    ]
    random.shuffle(positions)
    for color_name, (center_x, center_y) in zip(assignments, positions):
        half_size = random.randint(42, 51)
        box = (
            center_x - half_size,
            center_y - half_size,
            center_x + half_size,
            center_y + half_size,
        )
        if random.choice(("circle", "square")) == "circle":
            draw.ellipse(box, fill=SHAPE_COLORS[color_name], outline=(35, 39, 44), width=3)
        else:
            draw.rectangle(box, fill=SHAPE_COLORS[color_name], outline=(35, 39, 44), width=3)

    item = make_item(
        "count",
        index,
        f"How many {asked_color} shapes are there?",
        str(asked_count),
        f"There are {total} separated circles or squares; query targets {asked_color}.",
        (NUMBER_WORDS[asked_count],),
    )
    return image, item


WORD_SPECS = (
    (
        "URGENT",
        "the quiet workshop beside the harbor keeps a careful journal of invented machines and patient experiments. each afternoon, the team sorts copper pieces, checks painted gauges, and marks one {target} reminder beside the final sketch. before sunset, they pack every tool, sweep the long tables, and leave a friendly note for tomorrow's curious visitors.",
    ),
    (
        "HIDDEN",
        "the community garden behind the station grows herbs, beans, and cheerful rows of tiny flowers. during the weekly walk, volunteers compare leaf shapes, refill shallow watering cans, and place one {target} marker near the winding stone path. later, everyone shares warm bread, records the weather, and plans gentle tasks for the next bright morning.",
    ),
    (
        "BRIGHT",
        "the corner bakery tests imaginary recipes for travelers who enjoy unusual afternoon snacks. while the ovens warm, each baker measures flour, folds soft dough, and pins one {target} card above the cooling shelf. after the final tray arrives, the crew cleans every bowl, labels the paper bags, and invents a playful flavor for the coming weekend.",
    ),
    (
        "SILENT",
        "the little library near the meadow maintains a collection of maps for places that never existed. on rainy evenings, readers trace river lines, compare faded symbols, and leave one {target} clue inside the shared notebook. before closing, the librarian straightens each chair, dims the reading lamps, and prepares a fresh puzzle for tomorrow's thoughtful guests.",
    ),
)


def wrap_text(draw, text, font, maximum_width):
    lines = []
    current = []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and draw.textlength(candidate, font=font) > maximum_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def generate_word(index, font_used):
    seed_image("word", index)
    target, template = WORD_SPECS[index - 1]
    paragraph = template.format(target=target)
    words = paragraph.split()
    capitals = [
        token.strip(string.punctuation)
        for token in words
        if token.strip(string.punctuation).isalpha()
        and token.strip(string.punctuation).isupper()
    ]
    if not 40 <= len(words) <= 70:
        raise ValueError(f"word paragraph has {len(words)} words")
    if capitals != [target]:
        raise ValueError(f"word paragraph capitals are {capitals!r}")

    backgrounds = (
        (255, 252, 239),
        (241, 250, 244),
        (248, 245, 255),
        (240, 248, 252),
    )
    image = Image.new("RGB", (WIDTH, HEIGHT), backgrounds[index - 1])
    draw = ImageDraw.Draw(image)
    heading_font = load_font(font_used, 28)
    paragraph_font = load_font(font_used, 30)
    draw.text((44, 24), "field note", font=heading_font, fill=(91, 84, 70))
    draw.line((44, 64, 676, 64), fill=(151, 143, 126), width=2)
    lines = wrap_text(draw, paragraph, paragraph_font, 632)
    wrapped = "\n".join(lines)
    bbox = draw.multiline_textbbox((44, 88), wrapped, font=paragraph_font, spacing=10)
    if bbox[2] > 680 or bbox[3] > 515:
        raise ValueError(f"wrapped paragraph does not fit: {bbox}")
    draw.multiline_text(
        (44, 88),
        wrapped,
        font=paragraph_font,
        fill=(31, 35, 40),
        spacing=10,
    )
    item = make_item(
        "word",
        index,
        "Which word is written in capital letters?",
        target,
        f"An original {len(words)}-word paragraph with exactly one all-caps word.",
    )
    return image, item


GENERATORS = {
    "bar": generate_bar,
    "table": generate_table,
    "screenshot": generate_screenshot,
    "digits": generate_digits,
    "count": generate_count,
    "word": generate_word,
}


def validate_items(items):
    if len(items) != 24:
        raise ValueError(f"expected 24 items, got {len(items)}")
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest IDs are not unique")
    for category in CATEGORIES:
        category_items = [item for item in items if item["category"] == category]
        if len(category_items) != 4:
            raise ValueError(f"expected four {category} items")
    for item in items:
        expected_image = f"{item['id']}.png"
        if item["image"] != expected_image:
            raise ValueError(f"image mismatch for {item['id']}")
        if not re.fullmatch(r"[a-z]+_[0-9]{2}", item["id"]):
            raise ValueError(f"invalid item ID: {item['id']}")
        if not item["answer"] or len(item["answer"].split()) > 2:
            raise ValueError(f"answer is not short: {item['id']}")
        if item["category"] == "digits":
            answer = item["answer"]
            if not 6 <= len(answer) <= 10 or any(char not in CODE_ALPHABET for char in answer):
                raise ValueError(f"invalid code alphabet: {answer}")


def render_readme(font_used, total_bytes):
    kibibytes = total_bytes / 1024
    lines = [
        "# Vision-Exp needle validation set",
        "",
        'This is a deterministic needle-in-image visual-QA validation set for the upcoming image-input ("Vision-Exp") feature.',
        "Its PASS band will be established from an UPSTREAM REFERENCE BINARY's own answers on these same items, not from a fixed absolute threshold here; 24/24 is not declared as the required bar.",
        "",
        f'Manifest schema: `{{"font_used": "...", "seed": {SEED}, "items": [...]}}` with one scored object per PNG.',
        f"Resolved font: `{font_used}`.",
        f"Measured `needle/` total: {total_bytes} bytes ({kibibytes:.1f} KiB).",
        "",
        "Regenerate from the repository root:",
        "```sh",
        "python3 tests/gen_vision_needle.py",
        "```",
        "Score JSONL replies, or run the built-in scorer self-test:",
        "```sh",
        "python3 tests/vision_needle_score.py path/to/replies.jsonl",
        "python3 tests/vision_needle_score.py --selftest",
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_size_stable_readme(output_dir, font_used):
    readme_path = os.path.join(output_dir, "README.md")
    base_total = sum(
        entry.stat().st_size
        for entry in os.scandir(output_dir)
        if entry.is_file() and entry.name != "README.md"
    )
    stated_total = 0
    for _ in range(20):
        contents = render_readme(font_used, stated_total)
        computed_total = base_total + len(contents.encode("utf-8"))
        if computed_total == stated_total:
            with open(readme_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(contents)
            break
        stated_total = computed_total
    else:
        raise RuntimeError("README size statement did not converge")

    actual_total = sum(
        entry.stat().st_size for entry in os.scandir(output_dir) if entry.is_file()
    )
    if actual_total != stated_total:
        raise RuntimeError(
            f"README size statement is stale: stated {stated_total}, actual {actual_total}"
        )
    return actual_total


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    font_used = resolve_font()
    items = []

    for category in CATEGORIES:
        for index in range(1, 5):
            image, item = GENERATORS[category](index, font_used)
            save_palette_png(image, os.path.join(output_dir, item["image"]))
            items.append(item)

    validate_items(items)
    manifest = {
        "font_used": font_used,
        "seed": SEED,
        "items": items,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    total_bytes = write_size_stable_readme(output_dir, font_used)
    print(f"generated {len(items)} images in {output_dir}")
    print(f"font: {font_used}")
    print(f"needle directory bytes: {total_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
