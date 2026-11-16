You are the incident manager at Larkspur Home, an online furniture retailer. Below is the full post-incident review for INC-2291. The VP of Engineering, who was travelling that day, wants an executive summary.

Write that executive summary: 150 to 220 words of plain prose (no bullet points, no headings, no tables). It must tell the VP how long customers were affected, what actually caused it and why it was able to happen, the impact in customers, failed checkouts and money, whether anyone was charged incorrectly, and the most important follow-up actions. Use the final, corrected figures where the review revises earlier ones, and leave out anything that is not part of this incident.

---

# Post-incident review: INC-2291 — Checkout failures on 17 October 2024

**Status:** Final (revised 21 October 2024)
**Severity:** SEV-1
**Incident commander:** Tom Becker (Engineering Manager, Payments)
**Authors:** Tom Becker, Ana Sousa (Platform), Mateus Lin (SRE)
**Executive summary:** *To be written by the incident manager for the VP of Engineering.*

All times are UTC unless marked otherwise. The customer support log in Appendix B is kept in Pacific Daylight Time (PDT, UTC−7) and is quoted as recorded.

## 1. Background

Checkout on larkspurhome.com is handled by `checkout-api`, which calls `payments-gateway` to authorise the customer's card and then writes the order to the orders database (`orders-db`, a primary/replica PostgreSQL pair). Service-to-service calls inside our production clusters go through the service mesh, which requires mutual TLS (mTLS): every sidecar proxy presents a short-lived workload certificate, and every workload certificate is signed by the mesh's **intermediate certificate authority**, which is in turn signed by our offline root CA.

Workload certificates are renewed automatically every 24 hours. The intermediate certificate has a two-year lifetime and is renewed by `cert-rotator`, a scheduled job that issues a new intermediate 30 days before the current one expires. The intermediate in use on 17 October, `mesh-intermediate-2022`, was issued on 17 October 2022 at 14:12:07 and was valid until **17 October 2024 at 14:12:07**. Under normal operation `cert-rotator` would have replaced it on 17 September 2024.

Between 3 and 12 September 2024 the Platform team migrated the mesh control plane from the `prod-east-1` cluster to the new `prod-east-2` cluster (project MESH-114). To stop two control planes from issuing certificates at the same time, the team suspended the `cert-rotator` job on 3 September. The migration plan included a step to resume it after cut-over, but that step lived in a separate post-migration checklist that was never completed, so `cert-rotator` stayed suspended.

Certificate-expiry monitoring at Larkspur covers public-facing certificates only (via an external scanner that checks our websites and APIs). Internal certificates, including the mesh intermediate, were not monitored.

## 2. Timeline

| Time (UTC) | Event |
|---|---|
| 14:12:07 | `mesh-intermediate-2022` expires. New mTLS handshakes between `checkout-api` and `payments-gateway` start failing with `x509: certificate has expired or is not yet valid`. |
| 14:12 | Checkout success rate falls from its normal 97% to about 3%. The remaining successes use long-lived connections that were opened before expiry and had not yet been recycled. |
| 14:13 | First customer complaint arrives through live chat (logged as 07:13 PDT in Appendix B). |
| 14:14 | The synthetic checkout probe fails for the first time. |
| 14:19 | Alert `CheckoutSuccessRateLow` fires and pages the Payments on-call engineer. The alert requires five consecutive minutes below threshold. |
| 14:21 | `orders-db-1` raises a replication-lag alarm and the database automatically fails over to `orders-db-2`. The lag was caused by the nightly analytics export, which had started late. |
| 14:22 | Failover completes. Writes were unavailable for about 40 seconds. |
| 14:23 | Priya Raman (Payments on-call) acknowledges the page. |
| 14:26 | In the incident channel: "Looks like the DB failover — should clear once the new primary warms up." |
| 14:31 | Tom Becker declares a SEV-1 and takes incident command. |
| 14:38 | Database ruled out: the checkout error rate is unchanged after the failover, and `payments-gateway` logs show that no checkout requests are reaching it at all. |
| 14:41 | Status page updated: "Some customers are unable to complete checkout. We are investigating." |
| 14:44 | Ana Sousa (Platform) finds the `x509: certificate has expired` errors in the `checkout-api` sidecar logs. |
| 14:47 | Platform confirms the mesh intermediate has expired and that `cert-rotator` has been suspended since 3 September. |
| 14:51 | Break-glass procedure started: two Platform engineers issue a new intermediate (`mesh-intermediate-2024`) from the offline root. |
| 14:56 | New intermediate distributed; sidecars reload and request new workload certificates. |
| 14:59 | Checkout success rate back at the 97% baseline. **Customer impact ends.** |
| 15:20 | Status page marked resolved. |
| 15:30 | First impact estimate posted in the incident channel (see section 5). |
| 15:40 | `cert-rotator` resumed and verified against the new intermediate (Ana Sousa). |
| 16:10 | Marketing sends a recovery email to customers whose checkout failed. |

## 3. Detection

The first signal was a customer complaint at 14:13, a minute after expiry, followed by the synthetic probe at 14:14. Our paging alert waited for five full minutes below threshold and paged at 14:19, seven minutes after impact began. The page was acknowledged four minutes later at 14:23. The status page was first updated at 14:41, 29 minutes after customers were first affected, which is outside our 15-minute target for SEV-1 incidents.

## 4. Investigation and the database red herring

The first responders saw two things at once: checkout failures and, at 14:21, a database failover alarm. Because the failover alarm was the most visible event in the dashboards, the team spent roughly fifteen minutes (14:23 to 14:38) treating it as the probable cause and waiting for the new primary to "warm up".

The failover was real but unrelated. It was triggered by replication lag from the late-running analytics export and completed cleanly at 14:22 with about 40 seconds of write unavailability. It did not cause or worsen the checkout failures: by 14:21 checkouts were already failing at the mTLS handshake between `checkout-api` and `payments-gateway`, before any request could reach the database, and the error rate was the same before, during and after the failover. We have found no orders lost or corrupted by the failover.

The decisive clue was that `payments-gateway` was receiving no checkout traffic at all, which pointed at the connection between the two services rather than at either service. The sidecar logs then showed the expired-certificate errors.

## 5. Impact

**Duration.** Customers were affected from 14:12 to 14:59 UTC, 47 minutes.

**Failed checkouts.** The first estimate, posted at 15:30 on the day, was about 31,000 failed checkout requests, taken from load-balancer error counts. On 18 October the Data team revised this. The checkout page automatically retries a failed payment step up to two more times, so most customers generated several failed requests for one attempt, and some of the traffic came from bots. After de-duplicating retries and removing bot traffic, the corrected figure is **18,412 failed checkout attempts from 6,950 unique customers**. The 31,000 figure should no longer be used.

**Revenue.** Finance (21 October) puts the gross value of the carts in the failed attempts at $514,300. By 23:59 UTC on 17 October, 4,120 of the 6,950 affected customers had gone on to complete an order, many after the 16:10 recovery email. Finance estimates the **net lost revenue at $198,000**.

**Payments.** No customer was charged for a failed checkout and no one was charged twice. Because the TLS handshake failed before `checkout-api` could send anything to `payments-gateway`, no card authorisation was attempted for any failed checkout. No refunds are required.

**Support.** Customer support received 1,214 contacts related to the incident between 14:13 and 18:00 UTC. The median chat wait time peaked at 26 minutes.

**Other systems.** Browsing, search, account pages and order tracking were unaffected. Orders placed before 14:12 were fulfilled normally.

## 6. Root cause

The mesh intermediate certificate expired because the automated job that renews it, `cert-rotator`, had been suspended on 3 September 2024 for the MESH-114 control-plane migration and was never resumed. With the intermediate expired, sidecars could not establish new mTLS connections, and `checkout-api` could not reach `payments-gateway`.

Two things allowed the suspended job to go unnoticed for six weeks:

1. The step to resume `cert-rotator` was in a separate post-migration checklist rather than in the migration runbook itself, and nobody owned that checklist after cut-over.
2. Internal certificates had no expiry monitoring, so nothing warned us in the 30 days before expiry.

## 7. What went well

- Once the expired certificate was found (14:44), the break-glass procedure worked as designed and service was restored within fifteen minutes.
- The two-person rule for issuing an intermediate from the offline root was followed despite the pressure.
- The recovery email brought back a large share of the affected customers on the same day.

## 8. What went poorly

- The five-minute alert window delayed the page.
- Responders anchored on the coincidental database failover for about fifteen minutes.
- The status page was updated 29 minutes after impact began.
- A critical maintenance step depended on an unowned checklist.

## 9. Follow-up actions

| # | Action | Owner | Due | State |
|---|---|---|---|---|
| 1 | Resume `cert-rotator` and verify it against the new intermediate | Ana Sousa (Platform) | 17 Oct 2024 | Done (15:40 on the day) |
| 2 | Add expiry alerting for every internal CA and intermediate certificate at 30, 14 and 3 days before expiry | Mateus Lin (SRE) | 1 Nov 2024 | In progress |
| 3 | Move "resume suspended jobs" into the migration runbook template as a gating step, so a migration cannot be closed with jobs still suspended | Dana Whitfield (Platform) | 8 Nov 2024 | Not started |
| 4 | Shorten the `CheckoutSuccessRateLow` alert window from five minutes to two | Priya Raman (Payments) | 25 Oct 2024 | In progress |
| 5 | Add the `x509: certificate has expired` signature to the Payments runbook's first-response checklist | Priya Raman (Payments) | 25 Oct 2024 | In progress |
| 6 | Require a status page update within 10 minutes of a SEV-1 declaration | Lea Martin (Communications) | 15 Nov 2024 | Not started |

Actions 1 and 2 address the root cause directly and are the priority.

## Appendix A: Incident channel excerpts (UTC)

> **14:24 Priya Raman:** paged for CheckoutSuccessRateLow, looking. also seeing orders-db failover at 14:21
> **14:26 Kofi Mensah:** Looks like the DB failover — should clear once the new primary warms up.
> **14:29 Priya Raman:** new primary healthy, replication caught up, checkout still failing at the same rate
> **14:31 Tom Becker:** declaring SEV-1, I have command. Priya stay on payments, Kofi on DB
> **14:35 Kofi Mensah:** DB connections from checkout-api are normal, but there aren't many — feels like requests are dying before the DB
> **14:38 Tom Becker:** DB ruled out. payments-gateway sees zero checkout traffic. something between checkout-api and payments-gateway. pulling Platform in
> **14:44 Ana Sousa:** sidecar logs on checkout-api full of `x509: certificate has expired or is not yet valid`
> **14:47 Ana Sousa:** confirmed — mesh intermediate expired 14:12:07. cert-rotator suspended since Sept 3 (MESH-114). starting break-glass
> **14:56 Ana Sousa:** new intermediate out, sidecars reloading
> **14:59 Priya Raman:** checkout success back to 97%
> **15:30 Mateus Lin:** rough impact from LB 5xx: ~31k failed checkout requests. will refine tomorrow with dedup

## Appendix B: Customer support log excerpts (PDT, UTC−7)

> **07:13** Chat: customer reports "payment failed, please try again" on three attempts with two different cards.
> **07:20** Spike in chats mentioning checkout errors; supervisor alerted.
> **07:34** Support lead asks in #support-escalations whether there is a known issue.
> **07:41** Macro "Checkout issue under investigation" published after status page update.
> **07:59** Agents report customers succeeding at checkout again.
> **08:30** Macro updated: "Resolved. You were not charged for failed attempts."

## Appendix C: Other activity on 17 October (not part of this incident)

INC-2290, a separate SEV-3, affected product images earlier the same day: between 09:05 and 09:40 UTC our image CDN served thumbnails slowly in parts of Europe after a configuration push by the CDN provider. Checkout was not affected. INC-2290 has its own review and is listed here only because both incidents appear in the same day's on-call report.

A planned price-list update was deployed at 13:30 UTC as scheduled and was unrelated to INC-2291.
