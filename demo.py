from models.AlphaFreeRecDemo import AlphaFreeRecDemo
from models.base.parse import parse_args

import gdown
import torch
import os
import shutil
import sys
import json
import re
import unicodedata
import curses

# -----------------------------
# CLI UI helpers (no extra deps)
# -----------------------------
CSI = "\033["
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESET = CSI + "0m"
BOLD = CSI + "1m"
DIM = CSI + "2m"
ORANGE = CSI + "38;5;208m"   # orange tone (256-color)
BLUE = CSI + "38;5;111m"     # blue tone (256-color)

def _supports_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    return term not in ("", "dumb")

_USE_COLOR = _supports_ansi()

def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _USE_COLOR else text

def _clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def _boxed_title(msg: str) -> str:
    # Unicode box drawing (good terminal compatibility). Replace with +---+ if it breaks.
    inner = f" {msg} "
    top = "┌" + "─" * len(inner) + "┐"
    mid = "│" + inner + "│"
    bot = "└" + "─" * len(inner) + "┘"
    return "\n".join([top, mid, bot])

# Simple block ASCII art font (7 rows per character)

_LETTERS = {
    "A": [
        "  ###  ",
        " #   # ",
        "#     #",
        "#######",
        "#     #",
        "#     #",
        "#     #",
    ],
    "L": [
        "#      ",
        "#      ",
        "#      ",
        "#      ",
        "#      ",
        "#      ",
        "#######",
    ],
    "P": [
        "###### ",
        "#     #",
        "#     #",
        "###### ",
        "#      ",
        "#      ",
        "#      ",
    ],
    "H": [
        "#     #",
        "#     #",
        "#     #",
        "#######",
        "#     #",
        "#     #",
        "#     #",
    ],
    "F": [
        "#######",
        "#      ",
        "#      ",
        "#####  ",
        "#      ",
        "#      ",
        "#      ",
    ],
    "R": [
        "###### ",
        "#     #",
        "#     #",
        "###### ",
        "#   #  ",
        "#    # ",
        "#     #",
    ],
    "E": [
        "#######",
        "#      ",
        "#      ",
        "###### ",
        "#      ",
        "#      ",
        "#######",
    ],
    " ": [
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
        "   ",
    ],
}

ALPHAFREE_LOGO = "\n".join([
    "  ___  _       _          ______             ",
    " / _ \\| |     | |         |  ___|            ",
    "/ /_\\ \\ |_ __ | |__   __ _| |_ _ __ ___  ___ ",
    "|  _  | | '_ \\| '_ \\ / _` |  _| '__/ _ \\/ _ \\",
    "| | | | | |_) | | | | (_| | | | | |  __/  __/",
    "\\_| |_/_| .__/|_| |_|\\__,_\\_| |_|  \\___|\\___|",
    "        | |                                  ",
    "        |_|                                  ",
])

def _render_ascii(word: str, pad: int = 1) -> str:
    word = word.upper()
    height = 7
    lines = [""] * height
    for ch in word:
        pat = _LETTERS.get(ch, [" " * 7] * height)
        for i in range(height):
            lines[i] += pat[i] + (" " * pad)
    return "\n".join(lines)

def show_splash():
    _clear_screen()
    print(_c(_boxed_title("* Welcome to AlphaFree Demo *"), ORANGE))
    print()
    #print(_c(_render_ascii("ALPHAFREE"), ORANGE))
    print(_c(ALPHAFREE_LOGO, ORANGE))
    print()
    input(_c("Press Enter to continue:", BLUE))
    _clear_screen()

def _match_items_by_keyword(items, keyword: str):
    kw = keyword.strip().lower()
    if not kw:
        return []
    return [(idx, title) for idx, title in items if kw in (title or "").lower()]

def _curses_select_items(matches, selected_ids: set):
    """
    matches: list[(idx:int, title:str)]
    selected_ids: set[int] (persistent across searches)
    Controls:
      ↑/↓ : move
      PgUp/PgDn : page
      Space : toggle select
      Enter : confirm and exit
      q / Esc : exit (keep current selections)
    """
    if not matches:
        return selected_ids

    def _ui(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)

        pos = 0
        top = 0

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            body_h = max(1, h - 2)  # reserve 2 lines for header/help

            header = "Search results: ↑↓ move | Space select | Enter confirm | q/Esc quit"
            stdscr.addnstr(0, 0, header, w - 1)

            # scroll window
            if pos < top:
                top = pos
            elif pos >= top + body_h:
                top = pos - body_h + 1

            view = matches[top:top + body_h]

            for i, (idx, title) in enumerate(view):
                y = 1 + i
                checked = "[x]" if idx in selected_ids else "[ ]"
                line = f"{checked} {title}"
                if (top + i) == pos:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addnstr(y, 0, line, w - 1)
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addnstr(y, 0, line, w - 1)

            help_line = f"Selected: {len(selected_ids)}   Showing {top+1}-{min(top+body_h, len(matches))} / {len(matches)}"
            stdscr.addnstr(h - 1, 0, help_line, w - 1)

            key = stdscr.getch()

            if key in (curses.KEY_UP, ord('k')):
                pos = max(0, pos - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                pos = min(len(matches) - 1, pos + 1)
            elif key == curses.KEY_NPAGE:  # PageDown
                pos = min(len(matches) - 1, pos + body_h)
            elif key == curses.KEY_PPAGE:  # PageUp
                pos = max(0, pos - body_h)
            elif key == ord(' '):
                idx, _ = matches[pos]
                if idx in selected_ids:
                    selected_ids.remove(idx)
                else:
                    selected_ids.add(idx)
            elif key in (10, 13):  # Enter
                break
            elif key in (27, ord('q')):  # Esc / q
                break

    curses.wrapper(_ui)
    return selected_ids

def draw_prompt_screen():
    _clear_screen()
    print(_c(_boxed_title("* AlphaFree Demo"), ORANGE))
    print()
    print(_c("Type comma-separated item IDs (e.g., 40, 30) and press Enter.", BLUE))
    print(_c("Type 'exit' to quit.", BLUE))
    print()

# --- add this helper (optional but cleaner) ---
def draw_results_screen(consumed_display, recs_display, err_msg=""):
    _clear_screen()
    if consumed_display is None:
        # first screen (no results yet)
        print(_c(_boxed_title("* AlphaFree Demo"), ORANGE))
        print()
        print(_c("Enter comma-separated item IDs (e.g., 40, 30).", BLUE))
        print(_c("Type 'exit' to quit.", BLUE))
    else:
        # results screen
        print_two_columns(consumed_display, recs_display)
        print()
        print(_c("Enter comma-separated item IDs (e.g., 40, 30). Type 'exit' to quit.", BLUE))

    if err_msg:
        print(_c(err_msg, BLUE))
    print()  # keep one blank line before prompt

def _normalize_predictions(pred, topk: int = 20):
    """
    Normalize model.predict output to a list of top-k item ids,
    regardless of whether the input is a list, numpy array, or torch.Tensor.
    """
    # torch.Tensor
    if hasattr(torch, "is_tensor") and torch.is_tensor(pred):
        pred = pred.detach().cpu().tolist()

    # numpy array or similar
    if hasattr(pred, "tolist") and not isinstance(pred, list):
        try:
            pred = pred.tolist()
        except Exception:
            pass

    # handle nested list like (1, k)
    if isinstance(pred, list) and len(pred) == 1 and isinstance(pred[0], list):
        pred = pred[0]

    # if [(item, score), ...], extract item only
    if isinstance(pred, list) and pred and isinstance(pred[0], (tuple, list)) and len(pred[0]) >= 1:
        try:
            pred = [x[0] for x in pred]
        except Exception:
            pass

    # final output as list of ints
    out = []
    for x in (pred or [])[:topk]:
        try:
            out.append(int(x))
        except Exception:
            out.append(x)
    return out

def _display_width(s: str) -> int:
    """Estimate the display width in terminal (wide chars like CJK/emoji count as 2)."""
    s = ANSI_RE.sub("", str(s))
    w = 0
    for ch in s:
        # combining/format characters have zero width
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Me", "Cf"):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w

def _truncate_to_width(s: str, width: int, ellipsis: str = "…") -> str:
    """Truncate string to fit within display width, appending ellipsis if possible."""
    s = str(s)
    if width <= 0:
        return ""
    if _display_width(s) <= width:
        return s

    ell_w = _display_width(ellipsis)
    target = width - ell_w if width > ell_w else width

    out = []
    w = 0
    for ch in s:
        cw = 0
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Me", "Cf"):
            cw = 0
        else:
            cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1

        if w + cw > target:
            break
        out.append(ch)
        w += cw

    if width > ell_w:
        return "".join(out) + ellipsis
    return "".join(out)

def _pad_to_width(s: str, width: int) -> str:
    """Pad string with trailing spaces to exactly match the given display width."""
    s = _truncate_to_width(str(s), width)
    pad = width - _display_width(s)
    if pad > 0:
        s += " " * pad
    return s

def print_two_columns(consumed_items, recommended_items,
                      left_title="User interacted items",
                      right_title="AlphaFree Recommendation Top-20"):
    cols = shutil.get_terminal_size((120, 20)).columns

    # center separator
    sep = "  │  "
    sep_w = _display_width(sep)

    # Do NOT enforce a minimum column width (e.g. max(20, ...)).
    # Exceeding terminal width causes line wrapping and breaks the separator alignment.
    available = cols - sep_w
    if available < 2:
        # fallback for extremely narrow terminals
        sep = "|"
        sep_w = 1
        available = cols - sep_w

    col_w = max(1, available // 2)

    left_header_raw = f"{left_title} ({len(consumed_items)})"
    right_header_raw = f"{right_title} ({len(recommended_items)})"

    left_header = _pad_to_width(left_header_raw, col_w)
    right_header = _pad_to_width(right_header_raw, col_w)

    # header
    print(_c(left_header, ORANGE) + _c(sep, ORANGE) + _c(right_header, ORANGE))
    print(_c("─" * col_w, ORANGE) + _c(sep, ORANGE) + _c("─" * col_w, ORANGE))

    # content
    left_lines = [f"{i:>2}. {it}" for i, it in enumerate(consumed_items, 1)]
    right_lines = [f"{i:>2}. {it}" for i, it in enumerate(recommended_items, 1)]

    n = max(len(left_lines), len(right_lines), 1)
    left_lines += [""] * (n - len(left_lines))
    right_lines += [""] * (n - len(right_lines))

    for l, r in zip(left_lines, right_lines):
        l_cell = _pad_to_width(l, col_w)
        r_cell = _pad_to_width(r, col_w)
        print(l_cell + _c(sep, ORANGE) + r_cell)


if __name__ == '__main__':

    #download demo weights if not exist
    weight_file_path = "./weights/inference_demo/weights.pth.tar"
    if not os.path.exists(weight_file_path):
        os.makedirs("./weights/inference_demo", exist_ok=True)
        file_id = "1dAbq-vfolKsQMckrS86czIJIBJMXBmHg"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, weight_file_path, quiet=False)

    #load model
    args, special_args = parse_args()
    model = AlphaFreeRecDemo(args)
    checkpoint = torch.load(weight_file_path)
    state_dict = checkpoint['state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("mlp_origin.")}
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.cuda(args.cuda)

    #load item titles (interaction_idx -> title)
    title_path = args.data_path + args.dataset + '/item_info/item_title.json'
    item_title_map = {}
    with open(title_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            item_title_map[record['idx']] = record['title']

    show_splash()

    #inference loop
    model.eval()
    with torch.no_grad():
        consumed_display = None
        recs_display = None
        err_msg = ""

        # Build a flat list for searching: [(idx, title), ...]
items = sorted(item_title_map.items(), key=lambda x: x[0])  # idx order
selected_ids = set()

last_left = None   # list[str] (titles)
last_right = None  # list[str] (titles)
last_msg = ""

def draw_main_screen(left_titles, right_titles, msg=""):
    _clear_screen()
    print(_c(_boxed_title("* AlphaFree Demo"), ORANGE))
    print()
    print(_c("Commands:", BLUE))

    # Align command + description nicely
    cmd_w = 12  # width for command column (tweak if needed)
    def cmd_line(cmd, desc):
        return f"  {cmd:<{cmd_w}} {desc}"

    print(_c(cmd_line("init", "Clear the current selection."), BLUE))
    print(_c(cmd_line("search <keyword>", "Search item titles and select with ↑/↓ + Space, then Enter."), BLUE))
    print(_c(cmd_line("recommend", "Generate Top-20 recommendations for the selected items."), BLUE))
    print(_c(cmd_line("exit", "Quit the demo."), BLUE))
    print()

    if left_titles is not None and right_titles is not None:
        print_two_columns(left_titles, right_titles,
                  left_title=f"Selected items",
                  right_title="AlphaFree Recommendation Top-20")
        print()

    if msg:
        print(_c(msg, BLUE))
        print()

with torch.no_grad():
    while True:
        draw_main_screen(last_left, last_right, last_msg)
        last_msg = ""

        cmdline = input(_c("AlphaFree > ", BLUE)).strip()
        if not cmdline:
            continue

        parts = cmdline.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "exit":
            break

        elif cmd == "init":
            selected_ids.clear()
            last_left, last_right = None, None
            last_msg = "Selection cleared."

        elif cmd == "search":
            if not arg.strip():
                last_msg = "Usage: search <keyword>"
                continue

            matches = _match_items_by_keyword(items, arg)
            if not matches:
                last_msg = f"No items found containing '{arg}'."
                continue

            # Open TUI selector (arrow keys / spacebar)
            selected_ids = _curses_select_items(matches, selected_ids)
            last_msg = f"Selected items: {len(selected_ids)}"

        elif cmd == "recommend":
            if len(selected_ids) == 0:
                last_msg = "No selected items. Run: search <keyword> and select items first."
                continue

            consumed = sorted(list(selected_ids))
            preds = model.predict(consumed)
            recs = _normalize_predictions(preds, topk=20)

            # Titles only (as requested)
            left_titles = [item_title_map.get(i, "Unknown") for i in consumed]
            right_titles = [item_title_map.get(i, "Unknown") for i in recs]

            last_left, last_right = left_titles, right_titles

        else:
            last_msg = f"Unknown command: {cmd} (try: init / search <keyword> / recommend / exit)"
