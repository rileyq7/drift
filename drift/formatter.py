"""
Drift Formatter — Token-stream reformatter.

Walks the token stream and rewrites whitespace, indentation, and blank
lines into a canonical form. Comments and string contents preserved verbatim.

Used by `drift fmt`. Round-trip safety: format(format(x)) == format(x).
"""

from .lexer import TT, Token, lex


_OPEN = {TT.LBRACE, TT.LPAREN, TT.LBRACKET}
_CLOSE = {TT.RBRACE, TT.RPAREN, TT.RBRACKET}

_NO_SPACE_AFTER = {TT.LPAREN, TT.LBRACKET, TT.DOT}
_NO_SPACE_BEFORE = {TT.COMMA, TT.RPAREN, TT.RBRACKET, TT.COLON, TT.DOT}


def _reindent_comment(value: str, indent: int) -> list[str]:
    """Render a (possibly multi-line `{- ... -}`) comment token at `indent`.

    The lexer stores the comment's raw source text, continuation lines and
    all, so their *original* indentation is baked into `value`. Re-applying
    the current `indent` on top of that verbatim would grow with every
    format pass. Dedent continuation lines to their common leading
    whitespace first, then re-indent, so formatting twice matches formatting
    once.
    """
    lines = value.splitlines()
    if len(lines) <= 1:
        return [("  " * indent) + value]

    prefix = "  " * indent
    body_lines = lines[1:]
    non_blank = [l for l in body_lines if l.strip()]
    if non_blank:
        common = min(len(l) - len(l.lstrip()) for l in non_blank)
    else:
        common = 0

    out = [prefix + lines[0]]
    for line in body_lines:
        dedented = line[common:] if line.strip() else ""
        out.append((prefix + dedented) if dedented else "")
    return out


def format_source(source: str) -> str:
    tokens = lex(source)
    out: list[str] = []
    indent = 0
    line_toks: list[Token] = []

    def flush():
        nonlocal line_toks
        if not line_toks:
            return
        rendered = ("  " * indent) + _render_line(line_toks)
        out.append(rendered.rstrip())
        line_toks = []

    pending_blank_after_close = False
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]

        if t.type == TT.EOF:
            break

        if t.type == TT.NEWLINE:
            flush()
            i += 1
            continue

        if t.type == TT.COMMENT:
            flush()
            for line in _reindent_comment(t.value, indent):
                out.append(line)
            i += 1
            continue

        if t.type == TT.RBRACE:
            flush()
            indent = max(indent - 1, 0)
            out.append(("  " * indent) + "}")
            if indent == 0:
                pending_blank_after_close = True
            i += 1
            continue

        if t.type == TT.LBRACE:
            line_toks.append(t)
            flush()
            indent += 1
            i += 1
            continue

        # First non-trivia token after a top-level closing brace gets a
        # blank line before it.
        if pending_blank_after_close:
            if out and out[-1] != "":
                out.append("")
            pending_blank_after_close = False

        line_toks.append(t)
        i += 1

    flush()
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


_STRING_ESCAPE_OUT = {
    '\\': '\\\\',
    '"': '\\"',
    '\n': '\\n',
    '\t': '\\t',
    '\r': '\\r',
    '\0': '\\0',
}


def _escape_drift_string(value: str) -> str:
    """Re-escape a decoded string value back into Drift source form.

    The lexer stores the *decoded* text (escapes resolved), so rendering it
    verbatim would corrupt quotes/backslashes and break the idempotence
    invariant format(format(x)) == format(x). Escaping mirrors the lexer's
    decode table so a round-trip is stable.
    """
    return ''.join(_STRING_ESCAPE_OUT.get(ch, ch) for ch in value)


def _render_token(t: Token) -> str:
    if t.type == TT.STRING:
        return '"' + _escape_drift_string(t.value) + '"'
    return t.value


_TYPE_GENERIC_HEADS = {"list", "dict", "set", "tuple", "optional", "confident"}


def _is_type_open(prev: Token) -> bool:
    return (
        prev.type == TT.TYPE_IDENT
        or (prev.type == TT.IDENT and prev.value in _TYPE_GENERIC_HEADS)
    )


def _render_line(toks: list[Token]) -> str:
    parts: list[str] = []
    in_type_params = 0  # depth inside `<...>` type parameters
    for i, t in enumerate(toks):
        text = _render_token(t)
        if i == 0:
            parts.append(text)
            if t.type == TT.LANGLE:
                in_type_params += 1
            continue
        prev = toks[i - 1]
        sticky = False
        # `name(` — call/decl head
        if t.type == TT.LPAREN and prev.type in (TT.IDENT, TT.TYPE_IDENT):
            sticky = True
        # opening type param: `list<` / `Foo<`
        if t.type == TT.LANGLE and _is_type_open(prev):
            sticky = True
            in_type_params += 1
        # inside type params, suppress space before identifiers and closes.
        elif in_type_params > 0:
            if t.type == TT.RANGLE:
                sticky = True
                in_type_params -= 1
            elif prev.type == TT.LANGLE:
                sticky = True
            elif prev.type == TT.COMMA:
                # one space after comma
                pass
        if sticky or prev.type in _NO_SPACE_AFTER or t.type in _NO_SPACE_BEFORE:
            parts.append(text)
        else:
            parts.append(" " + text)
    return "".join(parts)
