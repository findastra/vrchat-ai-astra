"""Rename every "Mai" identifier and name in this codebase to AI Astra.

Dry run by default: prints a summary of every change. Pass --apply to write.
Run from the repo root:  python tools/rename_mai_to_ai_astra.py [--apply] [paths...]

Rules (a "word" is a run of letters, digits and underscores; "mai" only matches
as a whole word-part, so main/email/domain/maintain are never touched):

  Constants, env vars   MAI_STATE_DIR          -> AI_ASTRA_STATE_DIR
  snake_case, files     mai_phoenix_desktop    -> ai_astra_phoenix_desktop
  CamelCase             MaiBackendAPI          -> AIAstraBackendAPI
  camelCase             maiBridge              -> aiAstraBridge
  Snake with Title      Run_Mai                -> Run_AIAstra
  Text people read      Mai's brain            -> AI Astra's brain
  Header-style          X-Mai-Token            -> X-AIAstra-Token
  Self-name / prefixes  'mai', 'mai:invoke'    -> 'astra', 'astra:invoke'
  Folder paths          C:\\mai\\, /mai/       -> unchanged (Mai's root folder)

Files whose names contain a Mai word-part are renamed with git mv when the
file is tracked. Safe to re-run: already-renamed code has nothing left to match.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

SKIP_SUFFIXES = ('package-lock.json', '.lock', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.db', '.zip', '.exe', '.dll', '.pyc')
SKIP_DIRS = ('node_modules/', '.git/', '__pycache__/')
SELF = 'tools/rename_mai_to_ai_astra.py'

WORD = re.compile(r'[A-Za-z0-9_]+')
# Splits a word into parts: underscores, camelCase humps, acronyms, digits.
PART = re.compile(r'_+|[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')


def convert_part(part, word):
    """Replacement for one 'mai' part inside a compound word."""
    snake = '_' in word
    if part == 'MAI':
        return 'AI_ASTRA' if (snake or word.isupper()) else 'AIAstra'
    if part == 'mai':
        return 'ai_astra' if (snake or word == part) else 'aiAstra'
    if part == 'Mai':
        return 'AIAstra'
    return None  # odd casing like 'mAi': leave and report


def convert_word(word, before, after, after2=''):
    """Return the replacement for a whole word, or None to leave it."""
    if word.lower() == 'mai':
        # Folder paths such as C:\mai\ or /mai/ stay as they are.
        if before and before in '\\/' and (after in '\\/' or after == ''):
            return None
        if word == 'mai':
            return 'astra'
        # Glued to code punctuation (X-Mai-Token, obj.Mai, Mai.attr) vs. ending a sentence ("for Mai.").
        tight = bool(before and before in '-:.') or (after in (':', '.') and after2[1:2].isalnum())
        if word == 'Mai':
            return 'AIAstra' if tight else 'AI Astra'
        if word == 'MAI':
            return 'AI_ASTRA' if tight else 'AI ASTRA'
        return None
    parts = PART.findall(word)
    if not any(p.lower() == 'mai' for p in parts):
        return None
    out = []
    for p in parts:
        if p.lower() == 'mai':
            new = convert_part(p, word)
            if new is None:
                return None
            # camelCase word starting with 'mai' (maiBridge): keep the lower first letter.
            out.append(new)
        else:
            out.append(p)
    return ''.join(out)


def rename_text(text, counter):
    def repl(m):
        word = m.group(0)
        s, e = m.span()
        before = text[s - 1] if s > 0 else ''
        after = text[e] if e < len(text) else ''
        new = convert_word(word, before, after, text[e:e + 2])
        if new is None or new == word:
            return word
        counter[(word, new)] += 1
        return new
    return WORD.sub(repl, text)


def tracked_files(paths):
    out = subprocess.run(['git', 'ls-files', *paths], capture_output=True, text=True, check=True).stdout
    return [f for f in out.split('\n') if f]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--apply', action='store_true', help='Write changes (default is a dry run).')
    ap.add_argument('paths', nargs='*', help='Limit to these paths (default: whole repo).')
    args = ap.parse_args()

    counter = collections.Counter()
    changed_files = []
    renames = []
    for f in tracked_files(args.paths):
        if f == SELF or f.endswith(SKIP_SUFFIXES) or any(d in f for d in SKIP_DIRS):
            continue
        try:
            with open(f, encoding='utf-8-sig', newline='') as fh:
                text = fh.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        new_text = rename_text(text, counter)
        if new_text != text:
            changed_files.append(f)
            if args.apply:
                with open(f, 'w', encoding='utf-8', newline='') as fh:
                    fh.write(new_text)
        head, base = os.path.split(f)
        new_base = rename_text(base, collections.Counter())
        if new_base != base:
            renames.append((f, os.path.join(head, new_base).replace('\\', '/')))

    if args.apply:
        for old, new in renames:
            subprocess.run(['git', 'mv', old, new], check=True)

    mode = 'APPLIED' if args.apply else 'DRY RUN'
    print(f'{mode}: {sum(counter.values())} replacements in {len(changed_files)} files, {len(renames)} file renames')
    for (old, new), n in sorted(counter.items(), key=lambda kv: -kv[1]):
        print(f'  {n:4}  {old}  ->  {new}')
    for old, new in renames:
        print(f'  rename  {old}  ->  {new}')
    if not args.apply:
        print('Nothing written. Re-run with --apply.')


if __name__ == '__main__':
    sys.exit(main())
