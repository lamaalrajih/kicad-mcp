"""
Minimal S-expression parser/serializer for KiCad files.

KiCad files (.kicad_sch, .kicad_pcb, .kicad_sym, .kicad_pro is JSON) are
stored as S-expressions. This module provides a small, dependency-free
reader/writer sufficient for authoring schematics programmatically.

Leaves are represented as:
- ``Sym``  : a bare token (keyword, number, unquoted symbol) -- serialized as-is
- ``str``  : a quoted string -- serialized with surrounding quotes and escaping

Nodes are Python ``list`` objects whose first element is usually a ``Sym``
naming the node (e.g. ``Sym('symbol')``).
"""
from typing import List, Union

Node = Union[list, "Sym", str]


class Sym(str):
    """A bare (unquoted) S-expression token."""
    __slots__ = ()


def tokenize(text: str):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "(":
            tokens.append("(")
            i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        elif c == '"':
            # quoted string with backslash escapes
            i += 1
            buf = []
            while i < n:
                ch = text[i]
                if ch == "\\" and i + 1 < n:
                    nxt = text[i + 1]
                    mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
                    buf.append(mapping.get(nxt, nxt))
                    i += 2
                    continue
                if ch == '"':
                    i += 1
                    break
                buf.append(ch)
                i += 1
            tokens.append(("str", "".join(buf)))
        else:
            # bare atom until whitespace or paren
            start = i
            while i < n and text[i] not in ' \t\r\n()"':
                i += 1
            tokens.append(("sym", text[start:i]))
    return tokens


def parse(text: str) -> list:
    """Parse a full S-expression document into a nested list."""
    tokens = tokenize(text)
    pos = 0

    def parse_node():
        nonlocal pos
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            node = []
            while tokens[pos] != ")":
                node.append(parse_node())
            pos += 1  # consume ')'
            return node
        elif tok == ")":
            raise ValueError("Unexpected ')'")
        else:
            pos += 1
            kind, val = tok
            return Sym(val) if kind == "sym" else val

    # Skip to first '('
    while pos < len(tokens) and tokens[pos] != "(":
        pos += 1
    if pos >= len(tokens):
        raise ValueError("No S-expression found")
    return parse_node()


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def dumps(node: Node, indent: int = 0) -> str:
    """Serialize a node back to KiCad-style S-expression text.

    Indentation is cosmetic; KiCad and kicad-cli parse regardless of layout.
    """
    pad = "\t" * indent
    if isinstance(node, list):
        if not node:
            return pad + "()"
        # Decide inline vs multiline: inline when no child is a list.
        has_list_child = any(isinstance(c, list) for c in node)
        if not has_list_child:
            inner = " ".join(dumps(c, 0) for c in node)
            return f"{pad}({inner})"
        parts = [f"{pad}({dumps(node[0], 0)}"]
        for child in node[1:]:
            if isinstance(child, list):
                parts.append("\n" + dumps(child, indent + 1))
            else:
                parts.append(" " + dumps(child, 0))
        parts.append("\n" + pad + ")")
        return "".join(parts)
    elif isinstance(node, Sym):
        return str(node)
    else:
        return '"' + _escape(str(node)) + '"'


# ---- convenience helpers ----

def find(node: list, name: str):
    """Return the first child list whose head Sym equals ``name``."""
    for child in node:
        if isinstance(child, list) and child and str(child[0]) == name:
            return child
    return None


def find_all(node: list, name: str) -> List[list]:
    return [
        child
        for child in node
        if isinstance(child, list) and child and str(child[0]) == name
    ]
