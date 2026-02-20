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
    print(_c(_boxed_title("* Welcome to AlphaFree Demo"), ORANGE))
    print()
    print(_c(_render_ascii("ALPHAFREE"), ORANGE))
    print()
    input(_c("Press Enter to continue", BLUE))
    _clear_screen()

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

    #load item titles
    # item_idx in meta is the original (pre-remap) index, assigned sequentially by first appearance.
    # sorted(item_idx) order matches the embedding array order and the model's internal index.
    title_df_path = args.data_path + args.dataset + '/item_info/item_meta.json'
    with open(title_df_path, 'r') as f:
        item_meta = json.load(f)
    titles = item_meta['title']        # {row_key: title}
    item_idxs = item_meta['item_idx']  # {row_key: item_idx_int}
    idx_to_title = {int(raw_idx): titles[k] for k, raw_idx in item_idxs.items()}
    item_title_map = {model_i: idx_to_title[idx] for model_i, idx in enumerate(sorted(idx_to_title))}

    show_splash()

    #inference loop
    model.eval()
    with torch.no_grad():
        while True:
            raw = input("Enter interaction items (comma-separated, e.g., 40, 30) or type 'exit' to quit: ").strip()
            if raw.lower() == 'exit':
                break

            try:
                consumed = [int(x.strip()) for x in raw.split(',') if x.strip()]
                preds = model.predict(consumed)
                recs = _normalize_predictions(preds, topk=20)

                consumed_display = [f"{i} | {item_title_map[i]}" for i in consumed]
                recs_display = [f"{i} | {item_title_map[i]}" for i in recs]

                _clear_screen()
                print_two_columns(consumed_display, recs_display)
                print()
            except ValueError:
                print("Invalid input. Please enter a list of integers separated by commas.")
