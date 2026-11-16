# Routing scorecard

2000 simulated tasks (mix `standard`, seed 1), a price of $15 per failed attempt, 10% of later turns after a 5-60 minute pause, up to 3 attempts per task. Prices and quality scores from prices.json (2026-11-12T12:28:15+00:00). No model was called: outcomes are drawn from each model's published ability against each task's difficulty (docs/03-routing.md#how-we-measure-routing).

A failed attempt costs someone time: noticing it, asking again, waiting, cleaning up; here that time has a price per failed attempt, an assumption you can change (`--human-cost`). Totals count model spend plus failed attempts times that price: they're what the work costs, not time saved.

## All tasks

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Model spend, all | Time lost to failed attempts, all | Escalated | Escalation helped | Too high | Too low | Saves vs Sonnet (model spend) | Saves vs Opus (model spend) | Saves vs Sonnet in total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Always Claude Opus | 97.7% | $1.43 | $2.53 | $3.95 | $2,788.19 | $4,935 | 0.0% | – | 88.8% | 1.8% | -95.4% | 0.0% | -16.7% |
| Always Claude Sonnet | 97.4% | $0.7326 | $2.66 | $3.40 | $1,427.19 | $5,190 | 0.0% | – | 88.3% | 2.1% | 0.0% | 48.8% | 0.0% |
| Always the cheapest (deepseek-v4.1-flash) | 94.6% | $0.0197 | $4.15 | $4.17 | $37.26 | $7,845 | 0.0% | – | 25.4% | 5.2% | 97.4% | 98.7% | -19.1% |
| auto, model_choice threshold | 97.5% | $1.01 | $2.62 | $3.63 | $1,963.81 | $5,115 | 4.5% | 81.5% | 63.0% | 2.1% | -37.6% | 29.6% | -7.0% |
| auto, model_choice cost_per_success | 96.8% | $0.4008 | $3.07 | $3.47 | $775.49 | $5,940 | 4.3% | 79.3% | 37.2% | 3.1% | 45.7% | 72.2% | -1.5% |
| auto, model_choice expected_cost | 97.7% | $0.4046 | $2.67 | $3.08 | $790.53 | $5,220 | 7.5% | 78.3% | 63.1% | 2.7% | 44.6% | 71.7% | 9.2% |
| auto, threshold, escalating on evidence (example) | 97.6% | $1.10 | $2.52 | $3.62 | $2,149.82 | $4,920 | 10.7% | 65.1% | 63.4% | 1.7% | -50.6% | 22.9% | -6.8% |
| auto, expected_cost, with its cascade (one escalation) | 97.8% | $0.5399 | $2.54 | $3.08 | $1,056.00 | $4,965 | 12.1% | 61.7% | 70.9% | 2.1% | 26.0% | 62.1% | 9.0% |
| Oracle: the cheapest model that succeeds (lower bound) | 98.7% | $0.1718 | $1.85 | $2.03 | $339.19 | $3,660 | 0.0% | – | 0.0% | 0.0% | 76.2% | 87.8% | 39.6% |

## By task class

### coding-easy (559 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 100.0% | $0.3849 | $0.9392 | $1.32 | 93.7% | 0.0% | named 559 |
| Always Claude Sonnet | 100.0% | $0.1921 | $0.9392 | $1.13 | 93.7% | 0.0% | named 559 |
| Always the cheapest (deepseek-v4.1-flash) | 100.0% | $0.0037 | $0.9392 | $0.9429 | 3.8% | 0.0% | named 559 |
| auto, model_choice threshold | 100.0% | $0.0533 | $0.9392 | $0.9925 | 35.1% | 0.0% | coding:2 369, coding:3 106, coding:4 81, general:2 3 |
| auto, model_choice cost_per_success | 100.0% | $0.0242 | $0.9392 | $0.9634 | 16.6% | 0.0% | coding:2 369, coding:3 106, coding:4 81, general:2 3 |
| auto, model_choice expected_cost | 100.0% | $0.0266 | $0.9392 | $0.9657 | 38.6% | 0.0% | coding:2 369, coding:3 106, coding:4 81, general:2 3 |
| auto, threshold, escalating on evidence (example) | 100.0% | $0.0562 | $0.9392 | $0.9954 | 35.4% | 0.0% | coding:2 369, coding:3 106, coding:4 81, general:2 3 |
| auto, expected_cost, with its cascade (one escalation) | 100.0% | $0.0277 | $0.9392 | $0.9669 | 63.9% | 0.0% | coding:2 369, coding:3 106, coding:4 81, general:2 3 |
| Oracle: the cheapest model that succeeds (lower bound) | 100.0% | $0.0037 | $0.9392 | $0.9429 | 0.0% | 0.0% | named 559 |

### coding-medium (740 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 100.0% | $1.53 | $1.28 | $2.81 | 91.4% | 0.4% | named 740 |
| Always Claude Sonnet | 100.0% | $0.7621 | $1.26 | $2.02 | 91.5% | 0.3% | named 740 |
| Always the cheapest (deepseek-v4.1-flash) | 99.9% | $0.0176 | $1.54 | $1.56 | 4.9% | 1.9% | named 740 |
| auto, model_choice threshold | 100.0% | $0.3005 | $1.34 | $1.64 | 90.3% | 0.8% | coding:3 437, coding:4 296, general:2 7 |
| auto, model_choice cost_per_success | 100.0% | $0.2046 | $1.40 | $1.60 | 38.6% | 1.2% | coding:3 437, coding:4 296, general:2 7 |
| auto, model_choice expected_cost | 100.0% | $0.2183 | $1.34 | $1.56 | 90.3% | 0.8% | coding:3 437, coding:4 296, general:2 7 |
| auto, threshold, escalating on evidence (example) | 100.0% | $0.4210 | $1.26 | $1.68 | 90.4% | 0.3% | coding:3 437, coding:4 296, general:2 7 |
| auto, expected_cost, with its cascade (one escalation) | 100.0% | $0.2467 | $1.32 | $1.56 | 90.4% | 0.7% | coding:3 437, coding:4 296, general:2 7 |
| Oracle: the cheapest model that succeeds (lower bound) | 100.0% | $0.0232 | $1.22 | $1.24 | 0.0% | 0.0% | named 740 |

### coding-hard (128 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 76.6% | $11.57 | $20.36 | $31.92 | 51.6% | 14.1% | named 128 |
| Always Claude Sonnet | 75.0% | $6.07 | $21.41 | $27.48 | 48.4% | 15.6% | named 128 |
| Always the cheapest (deepseek-v4.1-flash) | 55.5% | $0.2460 | $43.73 | $43.98 | 0.0% | 31.2% | named 128 |
| auto, model_choice threshold | 81.2% | $13.29 | $16.44 | $29.73 | 48.4% | 9.4% | coding:3 11, coding:4 31, coding:5 86 |
| auto, model_choice cost_per_success | 79.7% | $4.95 | $17.79 | $22.74 | 41.4% | 11.7% | coding:3 11, coding:4 31, coding:5 86 |
| auto, model_choice expected_cost | 80.5% | $4.90 | $17.18 | $22.08 | 43.0% | 10.9% | coding:3 11, coding:4 31, coding:5 86 |
| auto, threshold, escalating on evidence (example) | 82.8% | $13.86 | $14.86 | $28.72 | 52.3% | 6.2% | coding:3 11, coding:4 31, coding:5 86 |
| auto, expected_cost, with its cascade (one escalation) | 82.0% | $6.58 | $15.71 | $22.29 | 46.1% | 8.6% | coding:3 11, coding:4 31, coding:5 86 |
| Oracle: the cheapest model that succeeds (lower bound) | 87.5% | $1.97 | $10.98 | $12.95 | 0.0% | 0.0% | named 128 |

### coding-very-hard (10 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 60.0% | $43.91 | $42.50 | $86.41 | 20.0% | 40.0% | named 10 |
| Always Claude Sonnet | 60.0% | $25.06 | $47.50 | $72.56 | 20.0% | 50.0% | named 10 |
| Always the cheapest (deepseek-v4.1-flash) | 30.0% | $1.36 | $120.00 | $121.36 | 0.0% | 60.0% | named 10 |
| auto, model_choice threshold | 60.0% | $54.12 | $40.00 | $94.12 | 30.0% | 30.0% | coding:3 1, coding:4 1, coding:5 8 |
| auto, model_choice cost_per_success | 60.0% | $17.46 | $37.50 | $54.96 | 20.0% | 30.0% | coding:3 1, coding:4 1, coding:5 8 |
| auto, model_choice expected_cost | 60.0% | $17.55 | $37.50 | $55.05 | 20.0% | 30.0% | coding:3 1, coding:4 1, coding:5 8 |
| auto, threshold, escalating on evidence (example) | 60.0% | $55.40 | $40.00 | $95.40 | 30.0% | 30.0% | coding:3 1, coding:4 1, coding:5 8 |
| auto, expected_cost, with its cascade (one escalation) | 60.0% | $26.89 | $35.00 | $61.89 | 20.0% | 20.0% | coding:3 1, coding:4 1, coding:5 8 |
| Oracle: the cheapest model that succeeds (lower bound) | 80.0% | $11.98 | $15.00 | $26.98 | 0.0% | 0.0% | named 10 |

### chat-short (159 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 100.0% | $0.0057 | $0.0000 | $0.0057 | 100.0% | 0.0% | named 159 |
| Always Claude Sonnet | 100.0% | $0.0029 | $0.0000 | $0.0029 | 100.0% | 0.0% | named 159 |
| Always the cheapest (deepseek-v4.1-flash) | 100.0% | $0.0001 | $0.0000 | $0.0001 | 100.0% | 0.0% | named 159 |
| auto, model_choice threshold | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | general:1 159 |
| auto, model_choice cost_per_success | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | general:1 159 |
| auto, model_choice expected_cost | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | general:1 159 |
| auto, threshold, escalating on evidence (example) | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | general:1 159 |
| auto, expected_cost, with its cascade (one escalation) | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | general:1 159 |
| Oracle: the cheapest model that succeeds (lower bound) | 100.0% | $2.6e-05 | $0.0000 | $2.6e-05 | 0.0% | 0.0% | named 159 |

### chat (278 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 100.0% | $0.0540 | $1.13 | $1.19 | 92.4% | 0.0% | named 278 |
| Always Claude Sonnet | 100.0% | $0.0270 | $1.13 | $1.16 | 92.4% | 0.0% | named 278 |
| Always the cheapest (deepseek-v4.1-flash) | 100.0% | $0.00071 | $1.19 | $1.19 | 91.4% | 0.0% | named 278 |
| auto, model_choice threshold | 100.0% | $0.0027 | $1.13 | $1.14 | 92.4% | 0.0% | general:2 240, general:3 38 |
| auto, model_choice cost_per_success | 100.0% | $0.00079 | $1.19 | $1.19 | 91.7% | 0.0% | general:2 240, general:3 38 |
| auto, model_choice expected_cost | 100.0% | $0.0022 | $1.13 | $1.14 | 92.4% | 0.0% | general:2 240, general:3 38 |
| auto, threshold, escalating on evidence (example) | 100.0% | $0.0027 | $1.13 | $1.14 | 92.4% | 0.0% | general:2 240, general:3 38 |
| auto, expected_cost, with its cascade (one escalation) | 100.0% | $0.0023 | $1.13 | $1.14 | 92.4% | 0.0% | general:2 240, general:3 38 |
| Oracle: the cheapest model that succeeds (lower bound) | 100.0% | $0.00044 | $1.13 | $1.13 | 0.0% | 0.0% | named 278 |

### analysis (126 tasks)

| Strategy | Success | Model spend /finished | Time lost /finished | Total /finished, counting failed attempts | Too high | Too low | Levels |
|---|---|---|---|---|---|---|---|
| Always Claude Opus | 89.7% | $0.2476 | $7.96 | $8.21 | 73.8% | 9.5% | named 126 |
| Always Claude Sonnet | 87.3% | $0.1344 | $9.82 | $9.95 | 67.5% | 12.7% | named 126 |
| Always the cheapest (deepseek-v4.1-flash) | 65.9% | $0.0053 | $28.73 | $28.74 | 31.0% | 35.7% | named 126 |
| auto, model_choice threshold | 82.5% | $0.0427 | $12.84 | $12.88 | 59.5% | 16.7% | general:2 45, general:3 81 |
| auto, model_choice cost_per_success | 72.2% | $0.0086 | $22.09 | $22.10 | 44.4% | 28.6% | general:2 45, general:3 81 |
| auto, model_choice expected_cost | 86.5% | $0.0331 | $12.80 | $12.83 | 50.8% | 24.6% | general:2 45, general:3 81 |
| auto, threshold, escalating on evidence (example) | 82.5% | $0.0427 | $12.84 | $12.88 | 59.5% | 16.7% | general:2 45, general:3 81 |
| auto, expected_cost, with its cascade (one escalation) | 86.5% | $0.0491 | $11.83 | $11.88 | 57.9% | 19.8% | general:2 45, general:3 81 |
| Oracle: the cheapest model that succeeds (lower bound) | 93.7% | $0.0319 | $4.83 | $4.86 | 0.0% | 0.0% | named 126 |

## How sure

90% intervals from a paired bootstrap over the tasks (400 resamples). A few long, hard tasks carry much of the spend, so totals move more than success rates.

| Strategy | Success | Model spend /finished | Total /finished, counting failed attempts | Saves vs Sonnet (model spend) |
|---|---|---|---|---|
| Always Claude Opus | 97.1% to 98.2% | $1.31 to $1.56 | $3.57 to $4.35 | -102.3% to -87.5% |
| Always Claude Sonnet | 96.8% to 98.0% | $0.6666 to $0.8011 | $3.08 to $3.76 | 0.0% to 0.0% |
| Always the cheapest (deepseek-v4.1-flash) | 93.8% to 95.3% | $0.0178 to $0.0217 | $3.74 to $4.64 | 97.3% to 97.5% |
| auto, model_choice threshold | 96.9% to 98.0% | $0.8344 to $1.20 | $3.20 to $4.07 | -52.9% to -21.3% |
| auto, model_choice cost_per_success | 96.1% to 97.3% | $0.3445 to $0.4627 | $3.12 to $3.86 | 41.2% to 50.8% |
| auto, model_choice expected_cost | 97.1% to 98.2% | $0.3485 to $0.4654 | $2.77 to $3.43 | 40.2% to 49.7% |
| auto, threshold, escalating on evidence (example) | 97.0% to 98.1% | $0.9231 to $1.29 | $3.20 to $4.07 | -64.9% to -35.3% |
| auto, expected_cost, with its cascade (one escalation) | 97.2% to 98.2% | $0.4452 to $0.6390 | $2.77 to $3.45 | 17.6% to 34.1% |
| Oracle: the cheapest model that succeeds (lower bound) | 98.2% to 99.1% | $0.1265 to $0.2285 | $1.78 to $2.28 | 70.0% to 82.1% |

## Sensitivity: the price per failed attempt

Total per finished task, counting failed attempts, at each price per failed attempt ($0 is model spend alone).

| Strategy | $0 | $5 | $15 | $50 |
|---|---|---|---|---|
| Always Claude Opus | $1.43 | $2.27 | $3.95 | $9.85 |
| Always Claude Sonnet | $0.7326 | $1.62 | $3.40 | $9.61 |
| Always the cheapest (deepseek-v4.1-flash) | $0.0197 | $1.40 | $4.17 | $13.84 |
| auto, model_choice threshold | $1.01 | $1.88 | $3.63 | $9.75 |
| auto, model_choice cost_per_success | $0.4008 | $1.42 | $3.47 | $10.63 |
| auto, model_choice expected_cost | $0.4046 | $1.30 | $3.08 | $9.31 |
| auto, threshold, escalating on evidence (example) | $1.10 | $1.94 | $3.62 | $9.50 |
| auto, expected_cost, with its cascade (one escalation) | $0.5399 | $1.39 | $3.08 | $9.00 |
| Oracle: the cheapest model that succeeds (lower bound) | $0.1718 | $0.7899 | $2.03 | $6.35 |

## Sensitivity: mix

400 tasks per value. Each cell: success, model spend per finished task, and the total per finished task counting failed attempts.

| Strategy | mix easier | mix standard | mix harder |
|---|---|---|---|
| Always Claude Sonnet | 98.8%, $0.4987, $2.40 | 98.8%, $0.6421, $2.88 | 90.8%, $2.09, $8.95 |
| auto, model_choice threshold | 99.5%, $0.5574, $2.18 | 99.0%, $0.8083, $2.89 | 93.0%, $3.52, $9.29 |
| auto, model_choice cost_per_success | 98.2%, $0.2507, $2.35 | 98.8%, $0.3144, $2.63 | 89.0%, $1.52, $9.19 |
| auto, model_choice expected_cost | 99.0%, $0.2548, $2.15 | 99.0%, $0.3247, $2.60 | 90.0%, $1.52, $8.69 |

## Sensitivity: pause share

400 tasks per value. Each cell: success, model spend per finished task, and the total per finished task counting failed attempts.

| Strategy | pause_share 0.0 | pause_share 0.1 | pause_share 0.3 |
|---|---|---|---|
| Always Claude Sonnet | 98.8%, $0.5044, $2.74 | 98.8%, $0.6421, $2.88 | 98.8%, $0.6583, $2.90 |
| auto, model_choice threshold | 99.0%, $0.6347, $2.72 | 99.0%, $0.8083, $2.89 | 99.0%, $0.9165, $3.00 |
| auto, model_choice cost_per_success | 98.8%, $0.2119, $2.53 | 98.8%, $0.3144, $2.63 | 98.8%, $0.5640, $2.88 |
| auto, model_choice expected_cost | 98.8%, $0.2200, $2.50 | 99.0%, $0.3247, $2.60 | 98.8%, $0.5877, $2.90 |

## Sensitivity: quality error

400 tasks per value. Each cell: success, model spend per finished task, and the total per finished task counting failed attempts.

| Strategy | quality_error 0.0 | quality_error 3.0 |
|---|---|---|
| Always Claude Sonnet | 98.8%, $0.6421, $2.88 | 98.8%, $0.6342, $2.84 |
| auto, model_choice threshold | 99.0%, $0.8083, $2.89 | 98.2%, $0.9006, $3.27 |
| auto, model_choice cost_per_success | 98.8%, $0.3144, $2.63 | 98.2%, $0.3433, $2.90 |
| auto, model_choice expected_cost | 99.0%, $0.3247, $2.60 | 98.2%, $0.3506, $2.83 |

## Calibration

Simulated first-attempt success against Epoch's curves, by model (coding ability).

| Model | Ability | Hard class | FrontierSWE (Epoch) | SWE-bench mix | SWE-bench Verified (fit) |
|---|---|---|---|---|---|
| claude-fable | 171.1 | 69% | 70% | 89% | 88% |
| gemini-carbon | 168.5 | 61% | 61% | 88% | 87% |
| gpt-terra | 167.7 | 58% | 58% | 88% | 86% |
| gpt-astra | 167.5 | 57% | 58% | 88% | 86% |
| claude-opus | 167.4 | 57% | 57% | 88% | 86% |
| claude-sonnet | 166.8 | 54% | 55% | 88% | 86% |
| gemini-argon | 164.8 | 47% | 47% | 87% | 84% |
| gpt-sol | 163.9 | 44% | 44% | 87% | 84% |
| claude-haiku | 163.0 | 41% | 41% | 86% | 83% |
| kimi-k3.1 | 161.4 | 35% | 35% | 86% | 82% |
| grok | 161.2 | 35% | 34% | 86% | 82% |
| deepseek-v4.1-flash | 161.0 | 34% | 34% | 86% | 82% |
| gpt-luna | 160.3 | 32% | 31% | 85% | 81% |
| kimi-k3 | 159.8 | 30% | 30% | 85% | 81% |
| glm-5.3 | 159.6 | 29% | 29% | 85% | 81% |
| deepseek-v4.1-flash-1017 | 159.4 | 29% | 29% | 85% | 81% |
| mistral-large-4 | 158.9 | 27% | 27% | 85% | 80% |
| muse-spark | 158.5 | 26% | 26% | 84% | 80% |
| gemini-flash | 157.9 | 24% | 24% | 84% | 79% |
| muse-spark-open | 155.4 | 18% | 18% | 83% | 77% |
| glm-5.3-flash | 155.3 | 18% | 18% | 83% | 77% |
| deepseek-v4-pro | 146.0 | 4% | 5% | 74% | 68% |
| gemini-flash-lite | 145.6 | 4% | 5% | 74% | 68% |
| gemma-4-31b | 144.3 | 3% | 4% | 72% | 66% |
| minimax-m3 | 143.5 | 3% | 4% | 71% | 65% |
| qwen3-coder | 143.5 | 2% | 4% | 71% | 65% |
| nemotron-3-ultra | 142.5 | 2% | 3% | 69% | 64% |
| gpt-oss-120b | 140.2 | 1% | 2% | 65% | 61% |
| gpt-oss-20b | 135.0 | 0% | 1% | 53% | 55% |
| llama-4-maverick | 132.1 | 0% | 1% | 45% | 51% |

