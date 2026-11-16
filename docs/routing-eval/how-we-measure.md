# How we measure routing

Text for the website. The details, every assumption and its source are in
[docs/03-routing.md](../03-routing.md#how-we-measure-routing); the latest numbers are in [scorecard.md](scorecard.md).

## Two ways to get routing wrong

A router can send work to a model that's too weak: the answer is wrong, and a person spends time finding out and asking
again. Or it can send work to a model that's stronger than it needs: the answer is right, and the bill is higher than it
had to be. Saving money and getting good answers pull against each other, so we measure both, together.

## Measuring before real traffic

Merit Order includes a routing simulator. It runs thousands of simulated tasks through the gateway's own routing code,
the same functions that route real requests, and prices every turn the way the providers bill it, prompt caching
included. No model is called and nothing is spent.

The tasks look like real work: coding-agent sessions from renaming a function to redesigning a service, with the tools,
turns, context growth and pauses that sessions have, and chat questions, writing and analysis. Some are worded
misleadingly, as people do ("quick fix" for something hard). Each task has a known difficulty, and each model's chance
of finishing it comes from public benchmark results: Epoch AI's ability scale, FrontierSWE and SWE-bench Verified. The
simulated difficulty is checked against those benchmarks: hard tasks are solved as often as each model solves
FrontierSWE.

For every way of routing (always one model, Merit Order's routing, and an oracle that knows which cheapest model would
have succeeded), we report:

- how many tasks get finished, and what each finished task costs;
- the time lost to failed attempts, and the total counting failed attempts. A failed attempt costs someone time: noticing
  it, asking again, waiting, cleaning up; here that time has a price per failed attempt, an assumption you can change;
- how often the choice was more model than needed, or less;
- how much it saves against always using one strong model.

## Measuring on real traffic

Simulated tasks can't tell you whether a level is right for your team's work. Merit Order can also explore: a small,
random share of task starts goes one level down or up, and the outcomes (tests passing or failing, corrections, retries)
say whether the level was right. The simulator can read that data back and score other routing policies on it,
weighting each task by how likely it was to be explored, again without spending anything.

## Guarding against regressions

Every change to Merit Order runs the simulator on a standard set of tasks. If the default routing gets more expensive
per finished task in total, counting failed attempts, or finishes fewer tasks, the change doesn't merge until someone decides,
on purpose, that the trade is worth it.

## What it can't tell

The simulator is as good as its assumptions. It trusts public benchmarks for how able each model is, so it can't catch a
model that does worse on your work than its scores say; it gives every model the same number of turns and the same
reply length; and it doesn't model rate limits or outages. Those are what the live measurement is for.
