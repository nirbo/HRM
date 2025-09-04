from __future__ import annotations

"""Helpers for parsing [REASON] blocks and handling simple tasks.

This module intentionally avoids heavy dependencies so it can be used in
lightweight local tests. Real HRM integration will replace the stubs here.
"""

import json
import math
import operator as op
from typing import Any, Dict


def parse_reason_content(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        obj = json.loads(content)
        assert isinstance(obj, dict)
        return obj
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid JSON in [REASON]: {exc}")


def _safe_eval_expr(expr: str) -> float:
    """Very small safe evaluator for +,-,*,/,**, parentheses and numbers.

    Not a general-purpose evaluator; intended only for quick smoke checks.
    """

    # Shunting-yard algorithm to avoid eval().
    tokens = _tokenize(expr)
    rpn = _to_rpn(tokens)
    return _eval_rpn(rpn)


def _tokenize(s: str):
    num = ''
    for ch in s.replace(' ', ''):
        if ch.isdigit() or ch == '.':
            num += ch
            continue
        if num:
            yield float(num)
            num = ''
        if ch in '+-*/()':
            yield ch
        else:
            raise ValueError(f"bad char: {ch}")
    if num:
        yield float(num)


def _to_rpn(tokens):
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}
    out, stack = [], []
    for t in tokens:
        if isinstance(t, float):
            out.append(t)
        elif t in prec:
            while stack and stack[-1] in prec and prec[stack[-1]] >= prec[t]:
                out.append(stack.pop())
            stack.append(t)
        elif t == '(':
            stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(':
                out.append(stack.pop())
            if not stack:
                raise ValueError('mismatched parens')
            stack.pop()
        else:
            raise ValueError(f"bad token: {t}")
    while stack:
        tt = stack.pop()
        if tt in '()':
            raise ValueError('mismatched parens')
        out.append(tt)
    return out


def _eval_rpn(tokens):
    ops = {'+': op.add, '-': op.sub, '*': op.mul, '/': op.truediv}
    st = []
    for t in tokens:
        if isinstance(t, float):
            st.append(t)
        else:
            b = st.pop(); a = st.pop()
            st.append(ops[t](a, b))
    if len(st) != 1:
        raise ValueError('bad expression')
    return st[0]


def handle_stub_task(obj: Dict[str, Any]) -> str:
    """Handle a tiny built-in task for smoke tests.

    Example [REASON] JSON:
    {"task":"calc", "expression": "2+2*(3-1)"}
    """

    task = obj.get('task')
    if task == 'calc':
        expr = obj.get('expression', '')
        val = _safe_eval_expr(str(expr))
        if abs(val - int(val)) < 1e-9:
            return str(int(val))
        return f"{val:.6g}"

    # Default fallback until HRM domains are wired.
    return f"<UNSUPPORTED:{task}>"

