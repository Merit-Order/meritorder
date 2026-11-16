You're working in Merit Order, an LLM gateway written in Python: its code in src/meritorder/, its tests in
tests/, its config in config/, and design docs in docs/. You can reach the repository only through the tools:
list_files, read_file and search to look around; edit_file, write_file, delete_file or apply_patch to change
files; shell for commands in a throwaway read-only copy (grep, python -c, ...); and run_tests (pytest; the tests
that need Redis or Postgres skip, since nothing else is running).

Make the change the issue below asks for. Fit it to the code that's there: follow its conventions, and keep
everything that works now working. When you finish, tests/ is replaced by a stricter version that checks what the
issue describes, through the names it gives, and every test that passes now must still pass. Docs and the
changelog aren't graded. Run the tests before you finish.

When you're done, reply with a short summary of the change.
