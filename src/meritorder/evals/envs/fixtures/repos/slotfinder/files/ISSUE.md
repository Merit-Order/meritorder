# free_slots offers busy time; busy_summary splits back-to-back meetings

Two reports from the scheduling bot this week:

1. Dana has an all-hands 09:00-12:00 and a 1:1 10:00-11:00 inside it. `free_slots` for her day returns
   11:00-12:00 as free, and the bot booked a customer call there.
2. Omar had meetings 09:00-10:00, 10:00-10:30 and 10:30-11:00. `busy_summary` reports 3 blocks with a
   longest block of 60 minutes; it should be 1 block of 120 minutes.

tests/test_slotfinder.py reproduces both. Several modules share the interval helpers, so please fix the
cause rather than the symptoms, and keep the public API as it is.
