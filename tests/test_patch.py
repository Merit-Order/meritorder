"""The apply_patch format OpenAI's models write, as the codebase suite applies it."""

import pytest

from meritorder.evals.envs.patch import PatchError, apply, parse

APP = "import os\n\n\ndef add(a, b):\n    return a - b\n\n\ndef double(x):\n    return x * 2\n"


def test_update_add_delete_and_move():
    files = {"app.py": APP, "old.py": "x = 1\n", "keep.py": "k = 1\n"}
    patch = """*** Begin Patch
*** Update File: app.py
@@ def add(a, b):
-    return a - b
+    return a + b
*** Add File: new.py
+Y = 2
+
+Z = 3
*** Delete File: old.py
*** Update File: keep.py
*** Move to: kept.py
@@
-k = 1
+k = 2
*** End Patch"""
    out, done = apply(files, patch)
    assert out["app.py"] == APP.replace("a - b", "a + b") and out["new.py"] == "Y = 2\n\nZ = 3\n"
    assert "old.py" not in out and "keep.py" not in out and out["kept.py"] == "k = 2\n"
    assert done == ["updated app.py", "added new.py", "deleted old.py", "updated keep.py and moved it to kept.py"]
    assert files["app.py"] == APP  # the input isn't changed


def test_hunks_apply_in_order_with_context_and_forgive_whitespace():
    text = "def f():\n    return 1\n\n\ndef g():\n    return 1\n"
    patch = "*** Begin Patch\n*** Update File: m.py\n@@ def g():\n-    return 1\n+    return 2\n*** End Patch"
    assert apply({"m.py": text}, patch)[0]["m.py"] == text.replace("def g():\n    return 1", "def g():\n    return 2")
    # Two hunks for the same text: the second applies after the first.
    two = "*** Begin Patch\n*** Update File: m.py\n@@\n-    return 1\n+    return 10\n@@\n-    return 1\n+    return 20\n*** End Patch"
    assert apply({"m.py": text}, two)[0]["m.py"] == "def f():\n    return 10\n\n\ndef g():\n    return 20\n"
    # Trailing spaces in the patch, and an empty context line without its leading space.
    loose = "*** Begin Patch\n*** Update File: m.py\n@@\n def f():   \n-    return 1\n+    return 3\n\n\n def g():\n*** End Patch"
    assert apply({"m.py": text}, loose)[0]["m.py"].startswith("def f():\n    return 3\n\n\ndef g():")


def test_end_of_file_and_pure_additions():
    text = "a\nb\na\n"
    eof = "*** Begin Patch\n*** Update File: t.txt\n@@\n-a\n+z\n*** End of File\n*** End Patch"
    assert apply({"t.txt": text}, eof)[0]["t.txt"] == "a\nb\nz\n"  # the last "a", not the first
    add = "*** Begin Patch\n*** Update File: t.txt\n@@ b\n+inserted\n*** End Patch"
    assert apply({"t.txt": text}, add)[0]["t.txt"] == "a\ninserted\nb\na\n"  # inserted at its anchor


def test_bad_patches_change_nothing_and_say_why():
    files = {"app.py": APP}
    with pytest.raises(PatchError, match="aren't in the file"):
        apply(files, "*** Begin Patch\n*** Update File: app.py\n@@\n-    return a * b\n+    return 0\n*** End Patch")
    with pytest.raises(PatchError, match="can't find the line after @@"):
        apply(files, "*** Begin Patch\n*** Update File: app.py\n@@ def nothing():\n-x\n+y\n*** End Patch")
    with pytest.raises(PatchError, match="already exists"):
        apply(files, "*** Begin Patch\n*** Add File: app.py\n+x\n*** End Patch")
    with pytest.raises(PatchError, match="starts with '\\+'"):
        apply(files, "*** Begin Patch\n*** Add File: b.py\nx = 1\n*** End Patch")
    with pytest.raises(PatchError, match="no such file to update"):
        apply(files, "*** Begin Patch\n*** Update File: nope.py\n@@\n-x\n+y\n*** End Patch")
    with pytest.raises(PatchError, match="no such file to delete"):
        apply(files, "*** Begin Patch\n*** Delete File: nope.py\n*** End Patch")
    with pytest.raises(PatchError, match="changes nothing"):
        apply(files, "*** Begin Patch\n*** End Patch")
    with pytest.raises(PatchError, match="expected"):
        apply(files, "*** Begin Patch\n--- a/app.py\n+++ b/app.py\n*** End Patch")
    with pytest.raises(PatchError, match="not a path"):
        apply(files, "*** Begin Patch\n*** Add File: ../x\n+x\n*** End Patch", norm=lambda p: None if p.startswith("..") else p)
    with pytest.raises(PatchError, match="no hunks"):
        parse("*** Begin Patch\n*** Update File: app.py\n*** End Patch")
    # One bad section in three: none of them applies.
    mixed = "*** Begin Patch\n*** Add File: b.py\n+ok\n*** Update File: app.py\n@@\n-missing\n+x\n*** End Patch"
    with pytest.raises(PatchError):
        apply(files, mixed)
    assert files == {"app.py": APP}


def test_without_begin_and_end_markers():
    out, _ = apply({"t.txt": "a\n"}, "*** Update File: t.txt\n@@\n-a\n+b")
    assert out["t.txt"] == "b\n"
