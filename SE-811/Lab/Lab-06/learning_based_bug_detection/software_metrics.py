"""Dependency-free, approximate file-level metrics for Java source code."""

from __future__ import annotations

import re
import math


DECISION_PATTERN = re.compile(r"\b(?:if|for|while|case|catch)\b|&&|\|\||\?")
DECLARATION_PATTERN = re.compile(
    r"(?:^|[;{}(,])\s*(?:final\s+)?"
    r"(?:[A-Za-z_$][\w$]*(?:\s*<[^;{}()]+>)?(?:\s*\[\s*\])*)\s+"
    r"([A-Za-z_$][\w$]*)\s*(?=[=,;)])",
    re.MULTILINE,
)
COMMA_DECLARATION_PATTERN = re.compile(r",\s*([A-Za-z_$][\w$]*)\s*(?=[=,;])")
METHOD_PATTERN = re.compile(
    # Keep the parameter span bounded to avoid pathological backtracking on
    # generated or malformed historical Java files.
    r"\b[A-Za-z_$][\w$<>\[\], ?]{0,120}\s+[A-Za-z_$][\w$]*"
    r"\s*\([^;{}\n]{0,500}\)\s*\{"
)
IMPORT_PATTERN = re.compile(r"\bimport\s+(?:static\s+)?([\w$]+(?:\.[\w$]+)*)\s*;")
NEW_PATTERN = re.compile(r"\bnew\s+([A-Z_$][\w$]*)")
TYPE_REFERENCE_PATTERN = re.compile(r"\b(?:extends|implements)\s+([A-Z_$][\w$]*)")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?")
OPERATOR_PATTERN = re.compile(
    r"===|!==|==|!=|<=|>=|\+\+|--|&&|\|\||\+=|-=|\*=|/=|%=|<<|>>|>>>|"
    r"[+*/%<>=!&|^~-]"
)
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "if", "implements", "import",
    "instanceof", "int", "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile",
    "while", "true", "false", "null", "var", "record", "sealed", "permits",
}


def strip_comments_and_literals(source: str) -> str:
    """Replace Java comments and literal contents while preserving newlines."""
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and following == "/":
                output.extend("  ")
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                output.extend("  ")
                index += 2
                state = "block_comment"
                continue
            if char in {'"', "'"}:
                output.append(" ")
                state = "string" if char == '"' else "char"
            else:
                output.append(char)
        elif state == "line_comment":
            if char == "\n":
                output.append("\n")
                state = "code"
            else:
                output.append(" ")
        elif state == "block_comment":
            if char == "*" and following == "/":
                output.extend("  ")
                index += 2
                state = "code"
                continue
            output.append("\n" if char == "\n" else " ")
        else:
            if char == "\\" and following:
                output.extend("  ")
                index += 2
                continue
            quote = '"' if state == "string" else "'"
            if char == quote:
                state = "code"
            output.append("\n" if char == "\n" else " ")
        index += 1
    return "".join(output)


def calculate_metrics(source: str) -> dict[str, int]:
    cleaned = strip_comments_and_literals(source)
    loc = sum(1 for line in cleaned.splitlines() if line.strip())
    cyclomatic_complexity = 1 + len(DECISION_PATTERN.findall(cleaned))

    # Count declaration names. This intentionally remains an explainable lexical
    # approximation; it includes fields, local variables, and parameters.
    variables = len(DECLARATION_PATTERN.findall(cleaned))
    for statement in cleaned.split(";"):
        if DECLARATION_PATTERN.search(statement):
            variables += len(COMMA_DECLARATION_PATTERN.findall(statement))

    return {
        "LOC": loc,
        "Cyclomatic_Complexity": cyclomatic_complexity,
        "Number_of_Variables": variables,
    }


def _halstead_metrics(cleaned: str) -> dict[str, float]:
    operators = OPERATOR_PATTERN.findall(cleaned)
    operands = [
        token for token in IDENTIFIER_PATTERN.findall(cleaned)
        if token not in JAVA_KEYWORDS
    ]
    n1, n2 = len(set(operators)), len(set(operands))
    N1, N2 = len(operators), len(operands)
    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocabulary) if vocabulary > 1 else 0.0
    difficulty = ((n1 / 2) * (N2 / n2)) if n2 else 0.0
    return {
        "Halstead_Volume": volume,
        "Halstead_Difficulty": difficulty,
        "Halstead_Effort": volume * difficulty,
    }


def _comment_density(source: str) -> float:
    total_lines = max(1, len(source.splitlines()))
    comment_lines = 0
    in_block = False
    for line in source.splitlines():
        text = line.strip()
        if in_block:
            comment_lines += 1
            if "*/" in text:
                in_block = False
            continue
        if text.startswith("//") or text.startswith("*"):
            comment_lines += 1
        if "/*" in text:
            comment_lines += 1
            in_block = "*/" not in text.split("/*", 1)[1]
        elif "//" in text:
            comment_lines += 1
    return comment_lines / total_lines


def _ck_metrics(cleaned: str) -> dict[str, int]:
    methods = list(METHOD_PATTERN.finditer(cleaned))
    method_count = len(methods)
    # WMC is approximated as one per method plus decision points in the file.
    wmc = method_count + len(DECISION_PATTERN.findall(cleaned))

    coupled: set[str] = set()
    coupled.update(name.rsplit(".", 1)[-1] for name in IMPORT_PATTERN.findall(cleaned))
    coupled.update(NEW_PATTERN.findall(cleaned))
    coupled.update(TYPE_REFERENCE_PATTERN.findall(cleaned))
    cbo = len(coupled)

    # CK LCOM approximation: compare identifiers used in each method body.
    method_fields: list[set[str]] = []
    for match in methods:
        start = match.end()
        depth = 1
        end = start
        while end < len(cleaned) and depth:
            if cleaned[end] == "{":
                depth += 1
            elif cleaned[end] == "}":
                depth -= 1
            end += 1
        identifiers = set(IDENTIFIER_PATTERN.findall(cleaned[start:end])) - JAVA_KEYWORDS
        method_fields.append(identifiers)
    shared = non_shared = 0
    for index, left in enumerate(method_fields):
        for right in method_fields[index + 1 :]:
            if left & right:
                shared += 1
            else:
                non_shared += 1
    return {"WMC": wmc, "CBO": cbo, "LCOM": max(0, non_shared - shared)}


def calculate_advanced_metrics(source: str) -> dict[str, float | int]:
    """Return original, product, and CK metrics for one Java file."""
    cleaned = strip_comments_and_literals(source)
    metrics: dict[str, float | int] = calculate_metrics(source)
    metrics.update(_halstead_metrics(cleaned))
    metrics["Comment_Density"] = _comment_density(source)
    metrics.update(_ck_metrics(cleaned))
    return metrics
