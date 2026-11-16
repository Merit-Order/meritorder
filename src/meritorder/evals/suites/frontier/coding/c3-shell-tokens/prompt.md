Write a Python module that splits a shell-like command line into words and operators. It must define:

```python
def tokenize(line: str) -> list[tuple[str, str]]: ...
```

It returns the tokens in order. Each token is a tuple `(kind, text)` where `kind` is `"word"` or `"op"`. Do not use the `shlex` module (importing it fails the task). The rules below are the whole specification; where they differ from a real shell, follow them.

## Characters outside quotes

- **Whitespace** is space, tab and newline (`" "`, `"\t"`, `"\n"`). Outside quotes it ends the current word. Runs of whitespace count as one separator, and leading or trailing whitespace produces nothing.
- **Operators**: `||`, `|`, `&&`, `&`, `;`, `>>`, `>`, `<`. Outside quotes, and when not escaped, these characters always form operator tokens, even with no whitespace around them: they end the current word and produce an `("op", ...)` token. Operators are matched greedily from left to right, longest first, so `>>>` is `>>` then `>`, and `|||` is `||` then `|`. A digit before `>` is an ordinary word character: `2>f` is the word `2`, the operator `>`, the word `f`.
- **Backslash** (`\`) outside quotes: if the next character is a newline, both are removed (a line continuation) and the current word continues as if they weren't there. Otherwise the backslash is removed and the next character is added to the current word literally, whatever it is (a space, a quote, an operator character, `#`, another backslash). A backslash as the last character of the input is an error.
- **Comments**: an unquoted, unescaped `#` that appears where a new token would start (at the start of the input, or right after whitespace or an operator) starts a comment, and everything up to, but not including, the next newline is ignored (the newline is then ordinary whitespace). A `#` anywhere else is an ordinary character: `a#b` is one word.
- Any other character is added to the current word.

## Quotes

- **Single quotes**: everything between `'` and the next `'` is added literally, including backslashes, `"`, `#`, newlines and operator characters. There are no escapes inside single quotes.
- **Double quotes**: characters between `"` and the next unescaped `"` are added literally, except for backslash: `\"`, `\\`, `\$` and `` \` `` add just the second character; a backslash followed by a newline removes both (a line continuation); a backslash followed by any other character is kept together with that character (`"\n"`, where `\n` here means a backslash and the letter n, stays as those two characters).
- Quotes don't end a word: quoted and unquoted parts that touch form one word (`a"b c"'d'e` is the single word `ab cde`).
- A word that contains quotes is produced even if it is empty: `''` and `""` each produce the word `""`; `a ''` produces two words, `a` and the empty string. But a line continuation alone doesn't create a word: `\` followed by a newline between two spaces produces nothing.

## Errors

Raise `ValueError` for an unterminated single or double quote and for a backslash at the very end of the input (outside single quotes; inside an unterminated double-quoted string, it's an unterminated quote, which is also `ValueError`). Text inside a comment is ignored, so quotes or a final backslash there are not errors.

## Examples

```python
tokenize("grep -n 'a b' file.txt|wc -l > out # count")
== [("word", "grep"), ("word", "-n"), ("word", "a b"), ("word", "file.txt"), ("op", "|"),
    ("word", "wc"), ("word", "-l"), ("op", ">"), ("word", "out")]

tokenize('echo "x\\"y" \\; a#b') == [("word", "echo"), ("word", 'x"y'), ("word", ";"), ("word", "a#b")]
tokenize("") == []
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library; the module must not print or do any work at import time.
