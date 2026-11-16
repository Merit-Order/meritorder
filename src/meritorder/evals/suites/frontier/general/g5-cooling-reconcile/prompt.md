You are the capacity planner at Northgate Data Centres. Today is 6 November 2024. A customer wants space in Hall C and you must give Sales a decision. The sources below disagree with each other in places: some are out of date, some are about other equipment, and some were later corrected. Work out which ones govern.

Answer three questions:

(a) Can Hall C accept Brightwave Analytics' full request (48 racks at 8 kW each) on the move-in date under current policy?
(b) If not, what is the maximum number of Brightwave's 8 kW racks that Hall C can accept on the move-in date?
(c) Would the phased plan in Source 10 let the remaining racks in later?

Show your calculation. Say which sources you relied on and which you set aside, and why. End your reply with a line `Final answer: <N>`, where N is your answer to (b) as a whole number of racks.

---

## Source 1 — Sales request CR-5521 (opened 1 November 2024)

**Customer:** Brightwave Analytics Ltd
**Requested space:** Hall C, Northgate London-2
**Requested load:** 48 racks, each drawing 8 kW of IT load (384 kW in total)
**Move-in date:** 13 January 2025
**Contract length:** 5 years
**Notes from Sales:** Brightwave's racks are standard 42U, front-to-back airflow, no liquid cooling. They have told us they want everything in one hall for latency reasons, so splitting across halls is not an option. Please confirm capacity by 8 November so we can send the order form.

## Source 2 — Northgate Cooling Capacity Policy, version 3.1

**Effective:** 1 July 2024. **Supersedes:** version 2.0 (March 2021).
**Owner:** Head of Critical Facilities

**1. Purpose.** This policy sets how cooling capacity is calculated when committing IT load to customers in any Northgate hall.

**2.1 Data source.** Use the manufacturer's net sensible cooling capacity for each installed unit model, taken from the manufacturer's *current* data sheet at the hall's return-air setpoint. Where a data sheet has been revised, only the latest revision may be used. Ratings from sales correspondence, marketing material or other models are not to be used.

**2.2 Setpoints.** Halls A, B and C operate at a return-air setpoint of 24 °C. Hall D, which has hot-aisle containment, operates at 27 °C. Capacity must be calculated at the hall's own setpoint; ratings at other temperatures may not be used, even if a unit could physically run warmer.

**2.3 Redundancy.** Hall A and Hall D are N+1. Hall B and Hall C are N+2. For an N+k hall, usable capacity equals the total net sensible capacity of all in-service units minus the capacity of the k largest in-service units.

**2.4 Non-IT heat.** Usable capacity must cover the IT load plus an allowance of 6% of the IT load for UPS losses, lighting and occupancy. (Version 2.0 used 5%; that figure is withdrawn.)

**2.5 Commitments.** A new commitment is allowed only if, on the move-in date, the hall's existing committed IT load plus the new IT load, plus the non-IT allowance on both, fits within usable capacity. Units that will be out of service on the move-in date count as zero capacity.

**2.6 Rounding.** Round rack counts down to whole racks. Do not round capacity figures until the final step.

**3. Exceptions.** Exceptions require written approval from the Head of Critical Facilities and are not available for new customer commitments.

## Source 3 — Internal wiki: "Hall C design notes"

*Last edited 14 June 2022 by J. Okoro.*

Hall C was fitted out in May 2020 with twelve Boreal CX-40 computer room air handlers (CRAHs), units C-01 to C-12, arranged along the two long walls. Raised floor depth is 900 mm; perforated tiles are 25% open in the cold aisles.

**Capacity.** Each CX-40 delivers 38.0 kW net sensible at our 24 °C return-air setpoint (Boreal data sheet). The hall is designed N+2, so usable capacity is 10 × 38.0 = 380 kW. After the standard 5% allowance for UPS losses and lighting, Hall C supports up to about 361 kW of IT load. In practice this means Hall C can take roughly 45 more racks at 8 kW.

**Chilled water.** Supply/return 14 °C / 20 °C from the shared plant (three chillers, N+1). The plant has spare capacity of about 1.2 MW and is not the constraint for Hall C.

**Power.** Two 1,600 kVA transformers feed Hall C via A and B UPS systems (each 600 kW). Power is not the constraint for Hall C at current densities.

**Known issues.** The tiles near C-05 and C-06 tend to under-deliver airflow; keep high-density racks away from that corner. The hall's fire suppression is inert gas (IG-55).

## Source 4 — Boreal CX-40 Product Data Sheet, Revision B (March 2021)

**Model:** CX-40 chilled-water computer room air handler, downflow, standard coil.

**Net sensible cooling capacity (kW), chilled water 14/20 °C, nominal airflow:**

| Return air temperature | 22 °C | 24 °C | 27 °C | 30 °C |
|---|---|---|---|---|
| Net sensible capacity | 33.5 | 38.0 | 43.5 | 48.0 |

**Airflow:** 9,800 m³/h nominal. **Fans:** 3 × AC centrifugal. **Fan power at nominal airflow:** 3.9 kW.
**Dimensions (W × D × H):** 1,750 × 890 × 1,980 mm. **Operating weight:** 610 kg.
**Sound pressure at 2 m:** 61 dB(A).
**Notes:** Capacities measured under the Boreal 2019 test procedure. Specifications subject to change without notice.

## Source 5 — Boreal CX-40 Product Data Sheet, Revision D (September 2023)

**This revision supersedes Revisions A, B and C.**

**Model:** CX-40 chilled-water computer room air handler, downflow, standard coil.

**Change history:**
- Rev C (June 2022): added an EC fan option.
- Rev D (September 2023): all capacities re-measured under the BTS-2023 test standard. Earlier revisions overstated net sensible capacity because fan heat was not fully deducted. **The re-measured figures apply to every CX-40 unit, whatever its production date or fan type.**

**Net sensible cooling capacity (kW), chilled water 14/20 °C, nominal airflow:**

| Return air temperature | 22 °C | 24 °C | 27 °C | 30 °C |
|---|---|---|---|---|
| Net sensible capacity | 30.0 | 34.0 | 40.0 | 45.5 |

**Airflow:** 9,800 m³/h nominal. **Fans:** 3 × AC centrifugal (EC optional). **Fan power at nominal airflow:** 3.9 kW (AC) / 2.7 kW (EC).
**Dimensions (W × D × H):** 1,750 × 890 × 1,980 mm. **Operating weight:** 610 kg.
**Sound pressure at 2 m:** 61 dB(A) (AC) / 58 dB(A) (EC).
**Note:** The CX-40-HD high-density variant (larger coil, 1,050 mm deep cabinet) is a separate model with its own data sheet. Standard CX-40 units cannot be converted to CX-40-HD.

## Source 6 — Boreal CX-60 Product Data Sheet, Revision A (May 2023)

*Revision A is the current and only revision of this data sheet.*

**Model:** CX-60 chilled-water computer room air handler, downflow.

**Net sensible cooling capacity (kW), chilled water 14/20 °C, nominal airflow:**

| Return air temperature | 22 °C | 24 °C | 27 °C | 30 °C |
|---|---|---|---|---|
| Net sensible capacity | 49.0 | 55.0 | 62.0 | 68.5 |

**Airflow:** 14,500 m³/h nominal. **Fans:** 4 × EC plug fans. **Fan power at nominal airflow:** 3.6 kW.
**Dimensions (W × D × H):** 2,150 × 890 × 1,980 mm. **Operating weight:** 790 kg.
**Sound pressure at 2 m:** 60 dB(A).
**Note:** Capacities measured under the BTS-2023 test standard.

## Source 7 — Email thread: "Hall C CRAH replacements and faults"

**From:** Márta Varga (Facilities Manager, London-2)
**To:** capacity-planning@northgate.example
**Date:** 12 February 2024
**Subject:** Hall C CRAH replacements

Hi all — heads-up that units C-11 and C-12 in Hall C are both being replaced with Boreal CX-60s in March. The old CX-40s are at end of life. The CX-60 is 55 kW net sensible at 24 °C per its data sheet, so Hall C should gain a fair bit of headroom. — Márta

**From:** Márta Varga
**Date:** 19 March 2024
**Subject:** RE: Hall C CRAH replacements

Correction to my February note: only C-11 has been swapped. The new CX-60 in position C-11 was commissioned on 18 March. The second CX-60 is back-ordered, so C-12 stays a CX-40 until the replacement arrives. Boreal now expects to deliver it in Q3 2025. — Márta

**From:** Rami Haddad (Capacity Planning)
**Date:** 2 April 2024
**Subject:** RE: Hall C CRAH replacements

Thanks Márta. DCIM updated: Hall C is now 11 × CX-40 plus 1 × CX-60 (C-11). — Rami

**From:** Márta Varga
**Date:** 15 October 2024
**Subject:** C-07 fault

C-07 tripped on a compressor fault this morning and is out of service until the replacement part arrives. — Márta

**From:** Márta Varga
**Date:** 28 October 2024
**Subject:** RE: C-07 fault

The part arrived early. C-07 was repaired and has been back in service since 14:00 today. No further work is planned on it. — Márta

**From:** Tuomas Lehto (Operations)
**Date:** 30 October 2024
**Subject:** Hall C setpoint

Could we raise Hall C's return-air setpoint to 27 °C? The data sheets show a lot more capacity at 27 °C. — Tuomas

**From:** Márta Varga
**Date:** 30 October 2024
**Subject:** RE: Hall C setpoint

Not without hot-aisle containment, which Hall C doesn't have, and the containment retrofit isn't funded. Policy keeps Hall C at 24 °C. — Márta

## Source 8 — Slack Connect message from Boreal's account representative (21 August 2024)

> **Dev Malhotra (Boreal):** Hi Rami! Following up on your question about getting more out of Hall C. Good news on the capacity front: the CX-40-HD (that's the high-density version with the bigger coil) is rated at 45 kW net sensible at 24 °C return air under BTS-2023. If you're planning more replacements it's worth a look. Happy to send a quote!

## Source 9 — DCIM export, London-2 cooling units (generated 4 November 2024)

| Hall | Unit | Model | Installed | Status |
|---|---|---|---|---|
| A | A-01 to A-08 | Boreal CX-40 (standard coil) | 2018-09 | In service |
| B | B-01 to B-10 | Boreal CX-60 | 2023-11 | In service |
| C | C-01 to C-06 | Boreal CX-40 (standard coil) | 2020-05 | In service |
| C | C-07 | Boreal CX-40 (standard coil) | 2020-05 | In service (compressor replaced 28 Oct 2024) |
| C | C-08 to C-10 | Boreal CX-40 (standard coil) | 2020-05 | In service |
| C | C-11 | Boreal CX-60 | 2024-03 | In service |
| C | C-12 | Boreal CX-40 (standard coil) | 2020-05 | In service; replacement with CX-60 expected Q3 2025 |
| D | D-01 to D-06 | Boreal CX-60 | 2022-02 | In service |

No maintenance outages are scheduled for Hall C between now and 31 March 2025.

**Committed IT load (existing contracts, including network core and meet-me room):**

| Hall | Committed IT load |
|---|---|
| A | 188.0 kW |
| B | 377.5 kW |
| C | 42.0 kW |
| D | 251.0 kW |

## Source 10 — Note from the Brightwave account manager (5 November 2024)

Brightwave would accept a phased move if we can't do all 48 racks in January: as many racks as we can give them on 13 January, and the rest once the second CX-60 goes into Hall C in Q3 2025. Can you confirm how many racks each phase would be? They'd like everything in Hall C by the end of 2025.
