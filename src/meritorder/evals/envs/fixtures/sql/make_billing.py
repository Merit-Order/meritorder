"""Generates billing.sql for a5-march-revenue (seeded, so the output is always the same; the output is committed).

    python make_billing.py billing.sql

Prints the expected answer (copy it into billing.yaml) and how much each trap moves it. Not imported by anything.
"""

import datetime as dt
import random
import sys
from decimal import ROUND_HALF_UP, Decimal

rng = random.Random(20260331)
OUT = sys.argv[1]

D = Decimal
EUR_PRICES = [D("19.90"), D("24.50"), D("49.00"), D("59.90"), D("89.00"), D("129.00"), D("249.00"), D("14.95")]
FOREIGN_PRICES = [D(20), D(45), D(60), D(99), D(150)]
TEST_IDS = {3, 17, 29, 41, 55}
CUTOVER = dt.datetime(2026, 3, 16)
DUAL_DAY = dt.date(2026, 3, 15)


def ts(y, m, d, H, M, S=0):
    return dt.datetime(y, m, d, H, M, S)


# ---------------------------------------------------------------- customers
customers = []
for cid in range(1, 61):
    if cid in TEST_IDS:
        is_test, email = 1, f"qa+{cid}@halyard.test"
    elif cid <= 14:
        is_test, email = None, f"customer{cid:03d}@example.com"
    else:
        is_test, email = 0, f"customer{cid:03d}@example.com"
    customers.append((cid, email, rng.choice(["PT", "ES", "GB", "US", "DE", "FR"]), is_test))

# ---------------------------------------------------------------- fx (business days; 2026-03-10 missing)
fx = {}
day = dt.date(2026, 2, 16)
while day <= dt.date(2026, 4, 3):
    if day.weekday() < 5 and day != dt.date(2026, 3, 10):
        fx[(day, "GBP")] = D(rng.choice(["1.14", "1.15", "1.16", "1.17", "1.18", "1.19"]))
        fx[(day, "USD")] = D(rng.choice(["0.90", "0.91", "0.92", "0.93", "0.94", "0.95"]))
    day += dt.timedelta(days=1)
# make sure neighbouring days differ where a weekend or the gap needs the previous rate
fx[(dt.date(2026, 2, 27), "GBP")], fx[(dt.date(2026, 3, 2), "GBP")] = D("1.16"), D("1.18")
fx[(dt.date(2026, 3, 9), "GBP")], fx[(dt.date(2026, 3, 11), "GBP")] = D("1.15"), D("1.17")
fx[(dt.date(2026, 3, 13), "USD")], fx[(dt.date(2026, 3, 16), "USD")] = D("0.91"), D("0.94")
fx[(dt.date(2026, 3, 20), "GBP")], fx[(dt.date(2026, 3, 23), "GBP")] = D("1.14"), D("1.19")
fx[(dt.date(2026, 3, 27), "USD")], fx[(dt.date(2026, 3, 30), "USD")] = D("0.92"), D("0.95")
fx[(dt.date(2026, 3, 31), "USD")], fx[(dt.date(2026, 4, 1), "USD")] = D("0.93"), D("0.90")


def rate(day, cur):
    if cur == "EUR":
        return D(1)
    d = day
    while (d, cur) not in fx:
        d -= dt.timedelta(days=1)
    return fx[(d, cur)]


# ---------------------------------------------------------------- logical payments
payments = []  # dicts: ref, order, cust, at, amount (major), cur, status
orders = []
next_order = [5001]


def new_order(cust, at):
    oid = next_order[0]
    next_order[0] += 1
    orders.append((oid, cust, (at - dt.timedelta(minutes=rng.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S")))
    return oid


def add(at, amount=None, cur=None, status="captured", cust="rand"):
    if cust == "rand":
        cust = None if rng.random() < 0.07 else rng.randint(1, 60)
    if cur is None:
        cur = rng.choices(["EUR", "GBP", "USD"], [0.74, 0.14, 0.12])[0]
    if amount is None:
        amount = rng.choice(EUR_PRICES) if cur == "EUR" else rng.choice(FOREIGN_PRICES)
    ref = "pay_" + "".join(rng.choice("0123456789abcdefghjkmnpqrstvwxyz") for _ in range(12))
    p = {"ref": ref, "order": new_order(cust, at), "cust": cust, "at": at, "amount": D(amount), "cur": cur, "status": status}
    payments.append(p)
    return p


start = dt.datetime(2026, 2, 20)
end = dt.datetime(2026, 4, 3, 12)
t = start
while t < end:
    t += dt.timedelta(minutes=rng.randint(150, 330))
    status = rng.choices(["captured", "failed", "voided"], [0.9, 0.07, 0.03])[0]
    add(t.replace(second=rng.randint(0, 59)), status=status)

# boundary and quirk rows
p_feb_edge = add(ts(2026, 2, 28, 23, 30, 12), D("129.00"), "EUR", cust=20)           # Feb in Lisbon
p_mar1 = add(ts(2026, 3, 1, 0, 20, 5), D(45), "GBP", cust=22)                       # Sunday: Friday's rate
p_gap = add(ts(2026, 3, 10, 11, 5, 40), D(150), "GBP", cust=8)                      # no rate on 03-10
p_sat = add(ts(2026, 3, 14, 16, 45, 0), D(99), "USD", cust=31)                      # Saturday
p_late_mar = add(ts(2026, 3, 31, 22, 30, 0), D("249.00"), "EUR", cust=36)           # March (text-compare trap)
p_apr1 = add(ts(2026, 3, 31, 23, 15, 0), D("89.00"), "EUR", cust=37)                # April in Lisbon
p_apr2 = add(ts(2026, 3, 31, 23, 50, 0), D(150), "USD", cust=38)                    # April in Lisbon
add(ts(2026, 3, 29, 0, 40, 0), D("59.90"), "EUR", cust=12)
add(ts(2026, 3, 29, 1, 30, 0), D(60), "GBP", cust=13)
for cust in (2, 5, 9, 11):  # legacy accounts (is_test NULL) buying in March
    add(ts(2026, 3, rng.randint(2, 28), rng.randint(8, 20), rng.randint(0, 59)), cust=cust)
for cust in TEST_IDS:  # internal test accounts
    add(ts(2026, 3, rng.randint(2, 28), rng.randint(8, 20), rng.randint(0, 59)), D("249.00"), "EUR", cust=cust)
guest = add(ts(2026, 3, 18, 13, 2, 0), D("129.00"), "EUR", cust=None)
for H in (9, 12, 15, 18):  # dual-write day
    add(ts(2026, 3, 15, H, rng.randint(0, 59), rng.randint(0, 59)))

payments.sort(key=lambda p: p["at"])

# ---------------------------------------------------------------- tables
legacy_rows, v2_rows = [], []
legacy_payments = [p for p in payments if p["at"] < CUTOVER]
v2_payments = [p for p in payments if p["at"].date() >= DUAL_DAY]
for p in legacy_payments:
    legacy_rows.append(p)
captured_legacy_march = [p for p in legacy_payments if p["status"] == "captured" and p["at"] >= dt.datetime(2026, 2, 26)]
dups = rng.sample(captured_legacy_march, 16) + [p_mar1]
legacy_eur_explicit = set(rng.sample([p["ref"] for p in legacy_payments if p["cur"] == "EUR"], 12))


def legacy_sql(rowid, p):
    cur = "NULL" if p["cur"] == "EUR" and p["ref"] not in legacy_eur_explicit else f"'{p['cur']}'"
    return (f"({rowid}, '{p['ref']}', {p['order']}, '{p['at'].strftime('%Y-%m-%d %H:%M:%S')}', {int(p['amount'] * 100)}, {cur}, "
            f"'{p['status'].upper()}')")


def v2_sql(rowid, p):
    amount = format(p["amount"].quantize(D("0.01")), "f")
    return f"({rowid}, '{p['ref']}', {p['order']}, '{p['at'].strftime('%Y-%m-%dT%H:%M:%SZ')}', {amount}, '{p['cur']}', '{p['status']}')"


legacy_lines, rowid = [], 1
dup_refs = {p["ref"] for p in dups}
pending = []
for p in legacy_rows:
    legacy_lines.append(legacy_sql(rowid, p))
    rowid += 1
    if p["ref"] in dup_refs:
        pending.append((rowid + rng.randint(1, 4), p))
    for due, q in [x for x in pending if x[0] <= rowid]:
        legacy_lines.append(legacy_sql(rowid, q))
        rowid += 1
        pending.remove((due, q))
for _, q in pending:
    legacy_lines.append(legacy_sql(rowid, q))
    rowid += 1

v2_lines = [v2_sql(i + 1, p) for i, p in enumerate(v2_payments)]

# ---------------------------------------------------------------- refunds
by_ref = {p["ref"]: p for p in payments}
cap = [p for p in payments if p["status"] == "captured"]


def pick(pred):
    choices = [p for p in cap if pred(p)]
    return rng.choice(choices)


refunds = []  # (ref, at, amount or None)
r_feb_paid = pick(lambda p: p["at"] < dt.datetime(2026, 2, 27) and p["cust"] not in TEST_IDS and p["cur"] == "EUR")
refunds.append((r_feb_paid["ref"], ts(2026, 3, 3, 10, 12, 0), None))                        # Feb payment, March refund (full)
r1 = pick(lambda p: dt.datetime(2026, 3, 2) <= p["at"] < dt.datetime(2026, 3, 9) and p["cur"] == "EUR" and p["cust"] not in TEST_IDS
          and p["amount"] > 40 and p["ref"] not in dup_refs)
refunds.append((r1["ref"], ts(2026, 3, 10, 9, 0, 0), D("20.00")))                             # partial
r_gbp = pick(lambda p: p["at"] >= dt.datetime(2026, 3, 16) and p["at"] < dt.datetime(2026, 3, 21) and p["cur"] == "GBP"
             and p["cust"] not in TEST_IDS)
refunds.append((r_gbp["ref"], ts(2026, 3, 21, 11, 30, 0), D(15)))                             # Saturday, GBP partial
r_test = next(p for p in payments if p["cust"] in TEST_IDS and p["status"] == "captured" and p["at"].month == 3)
refunds.append((r_test["ref"], ts(2026, 3, 24, 15, 0, 0), None))                              # test account: excluded
r_apr = pick(lambda p: p["at"] >= dt.datetime(2026, 3, 20) and p["at"] < dt.datetime(2026, 3, 30) and p["cust"] not in TEST_IDS
             and p["cur"] == "EUR")
refunds.append((r_apr["ref"], ts(2026, 3, 31, 23, 30, 0), None))                              # April in Lisbon
r_dup = next(p for p in dups if p["at"] >= dt.datetime(2026, 3, 1) and p["cust"] not in TEST_IDS and p["cust"] is not None)
refunds.append((r_dup["ref"], ts(2026, 3, 12, 14, 0, 0), None))                               # webhook-duplicated payment
r_feb = pick(lambda p: p["at"] < dt.datetime(2026, 2, 24) and p["cust"] not in TEST_IDS)
refunds.append((r_feb["ref"], ts(2026, 2, 25, 9, 0, 0), None))                                # February
r_usd = pick(lambda p: p["at"] >= dt.datetime(2026, 3, 16) and p["cur"] == "USD" and p["cust"] not in TEST_IDS
             and p["at"] < dt.datetime(2026, 3, 25))
refunds.append((r_usd["ref"], ts(2026, 3, 25, 16, 20, 0), None))                              # USD full
r_febedge = pick(lambda p: p["at"] < dt.datetime(2026, 2, 27) and p["cust"] not in TEST_IDS and p["ref"] != r_feb["ref"]
                 and p["ref"] != r_feb_paid["ref"])
refunds.append((r_febedge["ref"], ts(2026, 2, 28, 23, 45, 0), D("10.00") if r_febedge["cur"] == "EUR" else D(10)))  # Feb in Lisbon
refunds.append((guest["ref"], ts(2026, 3, 19, 8, 0, 0), D("29.00")))                          # guest order
r_null = next(p for p in payments if p["cust"] in (2, 5, 9, 11) and p["status"] == "captured" and p["at"].month == 3)
refunds.append((r_null["ref"], ts(2026, 3, 27, 12, 0, 0), None))                              # legacy account
r_later = pick(lambda p: p["at"] >= dt.datetime(2026, 3, 25) and p["at"] < dt.datetime(2026, 3, 31) and p["cust"] not in TEST_IDS
               and p["ref"] not in {r_apr["ref"]})
refunds.append((r_later["ref"], ts(2026, 4, 2, 10, 0, 0), None))                              # April
refunds.sort(key=lambda r: r[1])
refund_lines = [f"({i + 1}, '{ref}', '{at.strftime('%Y-%m-%d %H:%M:%S')}', {'NULL' if amt is None else format(amt, 'f')})"
                for i, (ref, at, amt) in enumerate(refunds)]

# ---------------------------------------------------------------- the answer, computed directly from the logical data
LIS_DST = dt.datetime(2026, 3, 29, 1, 0)


def lisbon(t):
    return t + dt.timedelta(hours=1 if t >= LIS_DST else 0)


def in_march(t):
    lt = lisbon(t)
    return (lt.year, lt.month) == (2026, 3)


def eur(amount, cur, day):
    return amount * rate(day, cur)


def answer(month=in_march, test_null_is_test=False, drop_guests=False, ignore_fx=False, refunds_on=True):
    def real(p):
        if p["cust"] is None:
            return not drop_guests
        flag = dict((c[0], c[3]) for c in customers)[p["cust"]]
        return flag == 0 or (flag is None and not test_null_is_test)

    revenue = sum((eur(p["amount"], p["cur"], p["at"].date()) if not ignore_fx else p["amount"])
                  for p in payments if p["status"] == "captured" and month(p["at"]) and real(p))
    ref_total = D(0)
    if refunds_on:
        for ref, at, amt in refunds:
            p = by_ref[ref]
            if month(at) and real(p):
                a = p["amount"] if amt is None else amt
                ref_total += eur(a, p["cur"], at.date()) if not ignore_fx else a
    return revenue - ref_total


truth = answer()
print("ANSWER", truth.quantize(D("0.01"), ROUND_HALF_UP))
print("utc month", answer(month=lambda t: (t.year, t.month) == (2026, 3)) - truth)
print("+1h all month", answer(month=lambda t: ((t + dt.timedelta(hours=1)).year, (t + dt.timedelta(hours=1)).month) == (2026, 3)) - truth)
print("is_test = 0", answer(test_null_is_test=True) - truth)
print("drop guests", answer(drop_guests=True) - truth)
print("ignore fx", answer(ignore_fx=True) - truth)
print("no refunds", answer(refunds_on=False) - truth)
print("rows legacy", len(legacy_lines), "v2", len(v2_lines), "refunds", len(refunds), "orders", len(orders))

SCHEMA = """\
-- Billing database snapshot, taken 2026-04-03. Read-only copy for analysis.

CREATE TABLE customers (
  -- One row per customer account.
  id INTEGER PRIMARY KEY,
  email TEXT NOT NULL,
  country TEXT,
  is_test INTEGER  -- 1 = internal test account (QA, demos), 0 = real customer. NULL for accounts created before the
                   -- flag existed (June 2025); those are all real customers.
);

CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  customer_id INTEGER REFERENCES customers(id),  -- NULL for guest checkouts (real customers without an account)
  created_at TEXT NOT NULL                       -- UTC, 'YYYY-MM-DD HH:MM:SS'
);

CREATE TABLE payments (
  -- LEGACY: written by the v1 webhook consumer. Frozen at the 2026-03-16 migration: payments from
  -- 2026-03-16 00:00 UTC on are only in payments_v2. Payments on 2026-03-15 (UTC) were written to
  -- both tables while the two consumers ran side by side.
  -- The v1 consumer stored a new row every time the provider retried a webhook delivery, so a
  -- provider_ref can appear on more than one row; each provider_ref is one payment.
  id INTEGER PRIMARY KEY,
  provider_ref TEXT NOT NULL,    -- the payment provider's id for the payment
  order_id INTEGER NOT NULL REFERENCES orders(id),
  event_at TEXT NOT NULL,        -- UTC, 'YYYY-MM-DD HH:MM:SS': when the payment was captured (or failed/voided)
  amount_cents INTEGER NOT NULL, -- minor units: cents, pence
  currency TEXT,                 -- ISO 4217; NULL means EUR (v1 left the field empty for EUR)
  status TEXT NOT NULL           -- 'CAPTURED', 'FAILED' or 'VOIDED'
);

CREATE TABLE payments_v2 (
  -- Written by the v2 consumer since the 2026-03-16 migration (it also has the 2026-03-15 payments,
  -- see payments). Webhook retries are deduplicated on insert: provider_ref is unique here.
  id INTEGER PRIMARY KEY,
  provider_ref TEXT NOT NULL UNIQUE,
  order_id INTEGER NOT NULL REFERENCES orders(id),
  event_at TEXT NOT NULL,        -- UTC, ISO 8601: 'YYYY-MM-DDTHH:MM:SSZ'
  amount NUMERIC NOT NULL,       -- major units: euros, pounds, dollars
  currency TEXT NOT NULL,        -- ISO 4217
  status TEXT NOT NULL           -- 'captured', 'failed' or 'voided'
);

CREATE TABLE refunds (
  -- Refunds issued to customers. A refund doesn't change the payment's row or status.
  id INTEGER PRIMARY KEY,
  provider_ref TEXT NOT NULL,    -- the refunded payment (payments.provider_ref or payments_v2.provider_ref)
  refunded_at TEXT NOT NULL,     -- UTC, 'YYYY-MM-DD HH:MM:SS'
  amount NUMERIC                 -- major units, in the payment's currency; NULL = the whole payment was refunded
);

CREATE TABLE fx_rates (
  -- Reference rates, published on business days only (no rows for weekends or days the feed missed).
  day TEXT NOT NULL,             -- 'YYYY-MM-DD'
  currency TEXT NOT NULL,        -- ISO 4217 (EUR itself isn't listed)
  eur_per_unit REAL NOT NULL,    -- 1 unit of currency = eur_per_unit EUR
  PRIMARY KEY (day, currency)
);
"""

with open(OUT, "w") as f:
    f.write(SCHEMA)
    f.write("\nINSERT INTO customers VALUES\n" + ",\n".join(
        f"({c[0]}, '{c[1]}', '{c[2]}', {'NULL' if c[3] is None else c[3]})" for c in customers) + ";\n")
    f.write("\nINSERT INTO orders VALUES\n" + ",\n".join(
        f"({o[0]}, {'NULL' if o[1] is None else o[1]}, '{o[2]}')" for o in sorted(orders)) + ";\n")
    f.write("\nINSERT INTO payments VALUES\n" + ",\n".join(legacy_lines) + ";\n")
    f.write("\nINSERT INTO payments_v2 VALUES\n" + ",\n".join(v2_lines) + ";\n")
    f.write("\nINSERT INTO refunds VALUES\n" + ",\n".join(refund_lines) + ";\n")
    f.write("\nINSERT INTO fx_rates VALUES\n" + ",\n".join(
        f"('{d.isoformat()}', '{c}', {r})" for (d, c), r in sorted(fx.items())) + ";\n")
