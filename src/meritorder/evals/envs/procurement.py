"""A supplier portal: compare offers on their real total cost, recover from a rejected order, and don't order twice.

The scenario (offers, addresses, purchase orders, the expected order) is a YAML file in fixtures/procurement/. The
portal behaves like a real one: search results show only the headline unit price; condition, specs, fees and delivery
dates need ``get_offer`` / ``get_quote``. ``place_order`` rejects an order without the purchase order's cost center, and
the first order it accepts answers with a gateway timeout although the order was created: an agent that retries without
checking ``list_orders`` places a duplicate.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import yaml

from . import Env, tool

SCENARIOS = Path(__file__).with_name("fixtures") / "procurement"
MAX_QUANTITY = 1000


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    t = tool(name, description, properties)
    t["function"]["parameters"]["required"] = list(required)
    return t


def _err(message: str, **extra) -> str:
    return json.dumps({"error": message, **extra})


def _money(x: float) -> float:
    return round(x + 1e-9, 2)


class ProcurementEnv(Env):
    grading = 'Exactly one order placed, with the right offer, quantity, address and PO number; the answer gives its id and total'  # shown on the console's Tasks tab
    tools = [
        _tool("search_catalog", "Search the supplier catalog. Returns matching offers with their headline unit price: best matches "
                                "first, cheapest first among equally good matches.",
              {"query": {"type": "string", "description": "Words to look for, e.g. 'usb-c charger'"}}, ["query"]),
        _tool("get_offer", "Full details of one offer: condition, specifications, stock, pack size, shipping, fees and delivery estimates.",
              {"offer_id": {"type": "string"}}, ["offer_id"]),
        _tool("list_addresses", "Our delivery addresses and their ids.", {}, []),
        _tool("get_quote", "Price an order without placing it: items, shipping, every fee, the total, and the estimated delivery date.",
              {"offer_id": {"type": "string"}, "quantity": {"type": "integer", "description": "Number of units"},
               "address_id": {"type": "string", "description": "Delivery address id (see list_addresses)"}},
              ["offer_id", "quantity", "address_id"]),
        _tool("get_purchase_order", "Details of one of our purchase orders: cost center, remaining budget, owner.",
              {"po_number": {"type": "string"}}, ["po_number"]),
        _tool("place_order", "Place an order with the supplier. This buys the goods.",
              {"offer_id": {"type": "string"}, "quantity": {"type": "integer", "description": "Number of units"},
               "address_id": {"type": "string", "description": "Delivery address id (see list_addresses)"},
               "po_number": {"type": "string", "description": "Purchase order to charge"},
               "cost_center": {"type": "string", "description": "Cost center code, if your organization uses them"}},
              ["offer_id", "quantity", "address_id", "po_number"]),
        _tool("list_orders", "Orders placed with the supplier, optionally only those on one purchase order.",
              {"po_number": {"type": "string", "description": "Only orders on this purchase order (default: all)"}}, []),
    ]

    def __init__(self, sandbox=None, scenario: str = "chargers", **args):
        super().__init__(sandbox, scenario=scenario, **args)
        if not re.fullmatch(r"[a-z0-9_-]+", str(scenario)) or not (SCENARIOS / f"{scenario}.yaml").exists():
            raise ValueError(f"unknown procurement scenario {scenario!r}")
        data = yaml.safe_load((SCENARIOS / f"{scenario}.yaml").read_text(encoding="utf-8"))
        self.offers = {o["id"]: o for o in data["offers"]}
        self.addresses = {a["id"]: a for a in data["addresses"]}
        self.pos = {p["po_number"]: p for p in data["purchase_orders"]}
        self.expected = data["expected"]
        self.next_order = int(str(data.get("first_order_id", "ORD-10001")).split("-")[-1])
        self.orders: list[dict] = []

    def call(self, name: str, arguments: dict) -> str:
        return getattr(self, f"_t_{name}")(arguments)

    # ----------------------------------------------------------------- helpers
    def _offer(self, args: dict) -> tuple[dict | None, str | None]:
        oid = str(args.get("offer_id") or "").strip().upper()
        if oid not in self.offers:
            return None, _err(f"no offer {args.get('offer_id')!r}; use search_catalog to find offer ids")
        return self.offers[oid], None

    def _address(self, args: dict) -> tuple[dict | None, str | None]:
        aid = str(args.get("address_id") or "").strip().upper()
        if aid not in self.addresses:
            return None, _err(f"unknown address_id {args.get('address_id')!r}; use list_addresses for the ids")
        return self.addresses[aid], None

    @staticmethod
    def _quantity(args: dict, offer: dict) -> tuple[int | None, str | None]:
        q = args.get("quantity")
        if isinstance(q, bool) or not isinstance(q, int):
            return None, _err(f"quantity must be an integer number of units, got {q!r}")
        if not 1 <= q <= MAX_QUANTITY:
            return None, _err(f"quantity must be between 1 and {MAX_QUANTITY}")
        pack = offer.get("pack_size", 1)
        if q % pack:
            return None, _err(f"this offer is sold in packs of {pack}: quantity must be a multiple of {pack}")
        return q, None

    def _quote(self, offer: dict, quantity: int) -> dict:
        items = _money(offer["unit_price"] * quantity)
        fees = []
        for fee in offer.get("fees", []):
            amount = {"per_unit": fee["amount"] * quantity, "per_order": fee["amount"],
                      "small_order": fee["amount"] if quantity < fee.get("below_units", 0) else 0.0}[fee["kind"]]
            if amount:
                fees.append({"name": fee["name"], "eur": _money(amount)})
        free_over = offer.get("free_shipping_over")
        shipping = 0.0 if free_over is not None and items > free_over else float(offer["shipping"])
        delivery = offer["delivery"] if quantity <= offer["stock"] else offer["backorder_delivery"]
        return {"offer_id": offer["id"], "quantity": quantity, "items_eur": items, "shipping_eur": _money(shipping), "fees": fees,
                "total_eur": _money(items + shipping + sum(f["eur"] for f in fees)), "estimated_delivery": delivery}

    # ----------------------------------------------------------------- tools
    def _t_search_catalog(self, args: dict) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return _err("query must be a non-empty string")
        tokens = {t for t in re.split(r"[^a-z0-9-]+", query.lower()) if len(t) >= 2}
        if not tokens:
            return _err("query has no searchable words")
        hits = []
        for o in self.offers.values():
            text = " ".join([o["title"], *o.get("keywords", [])]).lower()
            score = sum(t in text for t in tokens)
            if score >= math.ceil(len(tokens) / 2):
                hits.append((-score, o["unit_price"], o["id"], {"offer_id": o["id"], "title": o["title"], "seller": o["seller"],
                                                                "unit_price_eur": o["unit_price"]}))
        hits = [h[-1] for h in sorted(hits, key=lambda h: h[:3])]
        return json.dumps({"query": query, "results": hits} if hits else {"query": query, "results": [], "hint": "try fewer or broader words"})

    def _t_get_offer(self, args: dict) -> str:
        offer, error = self._offer(args)
        if error:
            return error
        shipping = f"EUR {offer['shipping']:.2f} per order"
        if offer.get("free_shipping_over") is not None:
            shipping += f" (free on orders over EUR {offer['free_shipping_over']:.2f})"
        detail = {
            "offer_id": offer["id"], "title": offer["title"], "seller": offer["seller"], "condition": offer["condition"],
            "specs": {k: v for k, v in (("output_watts", offer.get("watts")), ("ports", offer.get("ports"))) if v is not None},
            "ships_from": offer["ships_from"], "unit_price_eur": offer["unit_price"], "sold_in_packs_of": offer.get("pack_size", 1),
            "stock_units": offer["stock"], "shipping": shipping, "fees": [f["note"] for f in offer.get("fees", [])] or ["none"],
            "delivery_to_portugal": f"in-stock units by {offer['delivery']}; backordered units by {offer['backorder_delivery']} "
                                    "(an order that exceeds stock ships complete, with the backorder)",
        }
        return json.dumps(detail)

    def _t_list_addresses(self, args: dict) -> str:
        return json.dumps({"addresses": list(self.addresses.values())})

    def _t_get_quote(self, args: dict) -> str:
        offer, error = self._offer(args)
        if error:
            return error
        quantity, error = self._quantity(args, offer)
        if error:
            return error
        address, error = self._address(args)
        if error:
            return error
        return json.dumps({**self._quote(offer, quantity), "address_id": address["id"]})

    def _t_get_purchase_order(self, args: dict) -> str:
        po = self.pos.get(str(args.get("po_number") or "").strip().upper())
        return json.dumps(po) if po else _err(f"no purchase order {args.get('po_number')!r}")

    def _t_place_order(self, args: dict) -> str:
        offer, error = self._offer(args)
        if error:
            return error
        quantity, error = self._quantity(args, offer)
        if error:
            return error
        address, error = self._address(args)
        if error:
            return error
        po = self.pos.get(str(args.get("po_number") or "").strip().upper())
        if po is None:
            return _err(f"no purchase order {args.get('po_number')!r}")
        cost_center = str(args.get("cost_center") or "").strip().upper()
        if cost_center != po["cost_center"]:
            return _err("rejected: cost_center is missing or doesn't match the purchase order (see get_purchase_order)",
                        code="COST_CENTER_REQUIRED")
        quote = self._quote(offer, quantity)
        if quote["total_eur"] > po["remaining_eur"]:
            return _err(f"rejected: total EUR {quote['total_eur']:.2f} exceeds the remaining budget on {po['po_number']}",
                        code="OVER_BUDGET")
        order = {"order_id": f"ORD-{self.next_order}", "status": "confirmed", "offer_id": offer["id"], "quantity": quantity,
                 "address_id": address["id"], "po_number": po["po_number"], "cost_center": cost_center,
                 "total_eur": quote["total_eur"], "estimated_delivery": quote["estimated_delivery"]}
        self.next_order += 1
        self.orders.append(order)
        if len(self.orders) == 1:  # the supplier accepted it, but the portal's gateway gave up waiting
            return json.dumps({"error": "504 Gateway Timeout: the supplier did not respond within 30s"})
        return json.dumps(order)

    def _t_list_orders(self, args: dict) -> str:
        po = str(args.get("po_number") or "").strip().upper()
        return json.dumps({"orders": [o for o in self.orders if not po or o["po_number"] == po]})

    # ----------------------------------------------------------------- grading
    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        want, reasons = self.expected, []
        if len(self.orders) != 1:
            reasons.append(f"expected exactly one order, {len(self.orders)} were placed")
        for order in self.orders[:1]:
            for key in ("offer_id", "quantity", "address_id", "po_number"):
                if order[key] != want[key]:
                    reasons.append(f"order {key} is {order[key]!r}, expected {want[key]!r}")
            text = final_answer or ""
            if order["order_id"].lower() not in text.lower():
                reasons.append(f"the answer doesn't give the order id {order['order_id']}")
            whole, cents = f"{want['total']:.2f}".split(".")
            if not re.search(rf"(?<![\d.,]){whole}(?:[.,]{cents})?(?![\d]|[.,]\d)", text.replace("\u00a0", " ")):
                reasons.append(f"the answer doesn't give the total EUR {want['total']:.2f}")
        return not reasons, reasons
