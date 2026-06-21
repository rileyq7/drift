"""
Drift Lexer — Turns raw .drift source text into a stream of tokens.

This is the first stage of the transpiler pipeline:
    source text → [LEXER] → tokens → parser → AST → codegen → Python
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


class TT(Enum):
    """Token types. Kept minimal — the parser disambiguates by context."""
    # Literals
    STRING = auto()
    NUMBER = auto()
    CURRENCY = auto()
    DURATION = auto()
    BOOL = auto()

    # Identifiers
    IDENT = auto()         # snake_case: variable/step names
    TYPE_IDENT = auto()    # PascalCase: type/schema/agent names

    # Punctuation
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LANGLE = auto()        # <
    RANGLE = auto()        # >
    COMMA = auto()         # ,
    COLON = auto()         # :
    EQUALS = auto()        # =
    DOT = auto()           # .
    ARROW = auto()         # ->
    FAT_ARROW = auto()     # =>
    PIPE_ARROW = auto()    # |>
    TILDE_ARROW = auto()   # ~>
    LBRACKET = auto()     # [
    RBRACKET = auto()     # ]
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    GTE = auto()           # >=
    LTE = auto()           # <=
    EQEQ = auto()          # ==
    NEQ = auto()           # !=

    NEWLINE = auto()
    COMMENT = auto()       # -- line  or  {- block -}
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: str
    line: int
    col: int

    def __repr__(self):
        if self.type in (TT.NEWLINE, TT.EOF):
            return f"Token({self.type.name}, ln={self.line})"
        return f"Token({self.type.name}, {self.value!r}, ln={self.line})"


# Currency symbols we recognize
CURRENCY_SYMBOLS = {'£', '$', '€'}

# Two-character operators (checked before single-char)
TWO_CHAR_OPS = {
    '->': TT.ARROW,
    '=>': TT.FAT_ARROW,
    '|>': TT.PIPE_ARROW,
    '~>': TT.TILDE_ARROW,
    '>=': TT.GTE,
    '<=': TT.LTE,
    '==': TT.EQEQ,
    '!=': TT.NEQ,
}

SINGLE_CHAR_OPS = {
    '{': TT.LBRACE,
    '}': TT.RBRACE,
    '(': TT.LPAREN,
    ')': TT.RPAREN,
    '[': TT.LBRACKET,
    ']': TT.RBRACKET,
    '<': TT.LANGLE,
    '>': TT.RANGLE,
    ',': TT.COMMA,
    ':': TT.COLON,
    '=': TT.EQUALS,
    '.': TT.DOT,
    '+': TT.PLUS,
    '-': TT.MINUS,
    '*': TT.STAR,
    '/': TT.SLASH,
}


class LexError(Exception):
    def __init__(self, message: str, line: int, col: int):
        self.line = line
        self.col = col
        super().__init__(f"Line {line}, col {col}: {message}")


def lex(source: str) -> list[Token]:
    """Tokenize Drift source code into a list of tokens."""
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    length = len(source)

    while i < length:
        ch = source[i]

        # --- Skip whitespace (not newlines) ---
        if ch in (' ', '\t', '\r'):
            i += 1
            col += 1
            continue

        # --- Newlines ---
        if ch == '\n':
            # Collapse multiple newlines into one token
            if not tokens or tokens[-1].type != TT.NEWLINE:
                tokens.append(Token(TT.NEWLINE, '\\n', line, col))
            i += 1
            line += 1
            col = 1
            continue

        # --- Comments: -- to end of line ---
        if ch == '-' and i + 1 < length and source[i + 1] == '-':
            start = i
            start_col = col
            while i < length and source[i] != '\n':
                i += 1
            tokens.append(Token(TT.COMMENT, source[start:i], line, start_col))
            col = start_col + (i - start)
            continue

        # --- Block comments: {- ... -} ---
        if ch == '{' and i + 1 < length and source[i + 1] == '-':
            start = i
            start_line = line
            start_col = col
            i += 2
            col += 2
            depth = 1
            while i < length and depth > 0:
                if source[i] == '{' and i + 1 < length and source[i + 1] == '-':
                    depth += 1
                    i += 2
                    col += 2
                elif source[i] == '-' and i + 1 < length and source[i + 1] == '}':
                    depth -= 1
                    i += 2
                    col += 2
                elif source[i] == '\n':
                    i += 1
                    line += 1
                    col = 1
                else:
                    i += 1
                    col += 1
            tokens.append(Token(TT.COMMENT, source[start:i], start_line, start_col))
            continue

        # --- Strings: "..." with {expr} interpolation ---
        if ch == '"':
            start_line, start_col = line, col
            i += 1
            col += 1
            # Check for triple-quote
            if i + 1 < length and source[i] == '"' and source[i + 1] == '"':
                # Triple-quoted string
                i += 2
                col += 2
                buf = []
                while i < length:
                    if source[i] == '"' and i + 2 < length and source[i+1] == '"' and source[i+2] == '"':
                        i += 3
                        col += 3
                        break
                    if source[i] == '\n':
                        buf.append('\n')
                        line += 1
                        col = 1
                    else:
                        buf.append(source[i])
                        col += 1
                    i += 1
                tokens.append(Token(TT.STRING, ''.join(buf), start_line, start_col))
            else:
                # Regular string
                buf = []
                while i < length and source[i] != '"':
                    if source[i] == '\\' and i + 1 < length:
                        buf.append(source[i + 1])
                        i += 2
                        col += 2
                    elif source[i] == '\n':
                        raise LexError("Unterminated string", start_line, start_col)
                    else:
                        buf.append(source[i])
                        i += 1
                        col += 1
                if i >= length:
                    raise LexError("Unterminated string", start_line, start_col)
                i += 1  # skip closing "
                col += 1
                tokens.append(Token(TT.STRING, ''.join(buf), start_line, start_col))
            continue

        # --- Currency literals: £5, $0.10, €100 ---
        if ch in CURRENCY_SYMBOLS:
            start_col = col
            symbol = ch
            i += 1
            col += 1
            num_start = i
            while i < length and (source[i].isdigit() or source[i] == '.'):
                i += 1
                col += 1
            if i > num_start:
                tokens.append(Token(TT.CURRENCY, symbol + source[num_start:i], line, start_col))
            else:
                raise LexError(f"Expected number after currency symbol '{symbol}'", line, start_col)
            continue

        # --- Numbers (including duration: 30s, 5m, 2h, 1d) ---
        if ch.isdigit():
            start_col = col
            num_start = i
            while i < length and (source[i].isdigit() or source[i] == '.'):
                i += 1
                col += 1
            num_str = source[num_start:i]
            # Check for duration suffix
            if i < length and source[i] in ('s', 'm', 'h', 'd') and (i + 1 >= length or not source[i + 1].isalpha()):
                suffix = source[i]
                i += 1
                col += 1
                tokens.append(Token(TT.DURATION, num_str + suffix, line, start_col))
            else:
                tokens.append(Token(TT.NUMBER, num_str, line, start_col))
            continue

        # --- Two-character operators ---
        if i + 1 < length:
            two = source[i:i + 2]
            if two in TWO_CHAR_OPS:
                tokens.append(Token(TWO_CHAR_OPS[two], two, line, col))
                i += 2
                col += 2
                continue

        # --- Single-character operators ---
        if ch in SINGLE_CHAR_OPS:
            tokens.append(Token(SINGLE_CHAR_OPS[ch], ch, line, col))
            i += 1
            col += 1
            continue

        # --- Identifiers and keywords ---
        if ch.isalpha() or ch == '_':
            start_col = col
            start = i
            while i < length and (source[i].isalnum() or source[i] == '_'):
                i += 1
                col += 1
            word = source[start:i]

            # Boolean literals
            if word in ('true', 'false'):
                tokens.append(Token(TT.BOOL, word, line, start_col))
            # PascalCase = type identifier
            elif word[0].isupper():
                tokens.append(Token(TT.TYPE_IDENT, word, line, start_col))
            else:
                tokens.append(Token(TT.IDENT, word, line, start_col))
            continue

        raise LexError(f"Unexpected character: {ch!r}", line, col)

    # Always end with EOF
    tokens.append(Token(TT.EOF, '', line, col))
    return tokens
