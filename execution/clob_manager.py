"""Order management layer for Polymarket's CLOB API.

Wraps py-clob-client (v0.34.6) to provide maker-side operations: limit order
posting, cancellation, orderbook queries, and two-sided quoting. Operates in
dry-run mode when POLYMARKET_PRIVATE_KEY is not set, logging what it would do
without touching the API.

Designed for integration with the AvellanedaStoikov engine in
ml.strategies.market_making — call quotes_from_mm_engine() to bridge the two.
"""

import logging
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger(__name__)

POLYGON_CHAIN_ID = 137
DEFAULT_CLOB_URL = "https://clob.polymarket.com"
MIN_REQUEST_INTERVAL_SEC = 0.12  # ~8 req/s, well under 100/min CLOB limit


class CLOBManager:
    """Manages Polymarket CLOB API interactions for market making.

    Handles order creation, cancellation, and monitoring using py-clob-client.
    All methods are designed for maker-side operations (limit orders only).
    """

    def __init__(
        self,
        max_order_size: float = 500.0,
        max_total_exposure: float = 5000.0,
    ):
        self.host = os.getenv("POLYMARKET_CLOB_URL", DEFAULT_CLOB_URL)
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "")

        self.dry_run = not bool(private_key)
        self.max_order_size = max_order_size
        self.max_total_exposure = max_total_exposure
        self._last_request_time: float = 0.0
        self._total_exposure: float = 0.0

        if self.dry_run:
            logger.warning(
                "POLYMARKET_PRIVATE_KEY not set — running in DRY-RUN mode. "
                "No orders will be sent."
            )
            self.client = ClobClient(host=self.host, chain_id=POLYGON_CHAIN_ID)
        else:
            self.client = ClobClient(
                host=self.host,
                key=private_key,
                chain_id=POLYGON_CHAIN_ID,
            )
            self._derive_credentials()

    def _derive_credentials(self) -> None:
        """Derive and set API credentials from the private key."""
        try:
            creds = self.client.derive_api_key()
            self.client.set_api_creds(ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            ))
            logger.info("CLOB API credentials derived successfully")
        except Exception:
            logger.exception("Failed to derive CLOB API credentials — falling back to dry-run")
            self.dry_run = True

    # ── Rate limiting ────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Enforce minimum delay between API requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL_SEC:
            time.sleep(MIN_REQUEST_INTERVAL_SEC - elapsed)
        self._last_request_time = time.monotonic()

    # ── Size / exposure guards ───────────────────────────────────────

    def _check_order_limits(self, price: float, size: float) -> bool:
        """Return True if the order passes size and exposure limits."""
        notional = price * size
        if size > self.max_order_size:
            logger.warning(
                "Order rejected: size %.2f exceeds max_order_size %.2f",
                size, self.max_order_size,
            )
            return False
        if self._total_exposure + notional > self.max_total_exposure:
            logger.warning(
                "Order rejected: would push exposure to $%.2f (limit $%.2f)",
                self._total_exposure + notional, self.max_total_exposure,
            )
            return False
        return True

    # ── Order methods ────────────────────────────────────────────────

    def post_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> str | None:
        """Post a single limit order.

        Args:
            token_id: Polymarket condition token ID.
            side: "BUY" or "SELL".
            price: Limit price (0.01–0.99).
            size: Number of shares.
            order_type: "GTC" (good-til-cancelled) or "GTD".

        Returns:
            Order ID string on success, None on failure or dry-run.
        """
        side_const = BUY if side.upper() == "BUY" else SELL

        if not self._check_order_limits(price, size):
            return None

        if self.dry_run:
            logger.info(
                "[DRY-RUN] post_limit_order: %s %s %.0f @ %.4f (%s)",
                side, token_id[:12], size, price, order_type,
            )
            return None

        self._throttle()
        try:
            resp = self.client.create_and_post_order(OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side_const,
            ))
            order_id = resp.get("orderID") or resp.get("id")
            self._total_exposure += price * size
            logger.info(
                "Order posted: %s %s %.0f @ %.4f -> %s",
                side, token_id[:12], size, price, order_id,
            )
            return order_id
        except Exception:
            logger.exception("Failed to post order: %s %s @ %.4f", side, token_id[:12], price)
            return None

    def post_maker_quotes(
        self,
        token_id_yes: str,
        token_id_no: str,
        bid_price: float,
        bid_size: float,
        ask_price: float,
        ask_size: float,
    ) -> list[str]:
        """Post a two-sided quote: BUY YES at bid_price, SELL YES at ask_price.

        Args:
            token_id_yes: YES outcome token ID.
            token_id_no: NO outcome token ID (unused — quotes expressed on YES side).
            bid_price: Price to bid for YES tokens.
            bid_size: Number of YES tokens to bid for.
            ask_price: Price to offer YES tokens at.
            ask_size: Number of YES tokens to offer.

        Returns:
            List of order IDs for successfully posted legs.
        """
        order_ids: list[str] = []

        bid_id = self.post_limit_order(token_id_yes, "BUY", bid_price, bid_size)
        if bid_id:
            order_ids.append(bid_id)

        ask_id = self.post_limit_order(token_id_yes, "SELL", ask_price, ask_size)
        if ask_id:
            order_ids.append(ask_id)

        return order_ids

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID.

        Returns:
            True if cancelled (or dry-run), False on failure.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] cancel_order: %s", order_id)
            return True

        self._throttle()
        try:
            self.client.cancel(order_id=order_id)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel every open order across all markets.

        Returns:
            Number of orders cancelled.
        """
        if self.dry_run:
            logger.info("[DRY-RUN] cancel_all_orders")
            return 0

        self._throttle()
        try:
            resp = self.client.cancel_all()
            cancelled = len(resp) if isinstance(resp, list) else 1
            self._total_exposure = 0.0
            logger.info("Cancelled all orders (%d)", cancelled)
            return cancelled
        except Exception:
            logger.exception("Failed to cancel all orders")
            return 0

    def cancel_market_orders(self, token_id: str) -> int:
        """Cancel all open orders for a specific token.

        Returns:
            Number of orders cancelled for this token.
        """
        orders = self.get_open_orders(token_id)
        cancelled = 0
        for order in orders:
            oid = order.get("id") or order.get("orderID", "")
            if oid and self.cancel_order(oid):
                cancelled += 1
        return cancelled

    def get_open_orders(self, token_id: str | None = None) -> list[dict]:
        """Fetch open orders, optionally filtered by token.

        Works in both live and dry-run mode (returns empty list in dry-run
        since no real orders exist).
        """
        if self.dry_run:
            return []

        self._throttle()
        try:
            resp = self.client.get_orders()
            orders = resp if isinstance(resp, list) else []
            if token_id:
                orders = [
                    o for o in orders
                    if o.get("asset_id") == token_id or o.get("token_id") == token_id
                ]
            return orders
        except Exception:
            logger.exception("Failed to fetch open orders")
            return []

    # ── Market data (no auth required) ───────────────────────────────

    def get_orderbook(self, token_id: str) -> dict:
        """Fetch the full orderbook for a token.

        Returns:
            Dict with 'bids' and 'asks' lists, each entry having
            'price' and 'size'. Empty dict on failure.
        """
        self._throttle()
        try:
            book = self.client.get_order_book(token_id)
            return {
                "bids": getattr(book, "bids", []),
                "asks": getattr(book, "asks", []),
            }
        except Exception:
            logger.exception("Failed to fetch orderbook for %s", token_id[:12])
            return {"bids": [], "asks": []}

    def get_midpoint(self, token_id: str) -> float:
        """Return the mid price between best bid and best ask.

        Falls back to 0.5 if the book is empty.
        """
        spread_info = self.get_spread(token_id)
        if spread_info["best_bid"] is not None and spread_info["best_ask"] is not None:
            return (spread_info["best_bid"] + spread_info["best_ask"]) / 2.0
        return 0.5

    def get_spread(self, token_id: str) -> dict:
        """Compute best bid, best ask, and spread from the live orderbook.

        Returns:
            Dict with 'best_bid', 'best_ask', 'spread' (all float or None).
        """
        book = self.get_orderbook(token_id)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = None
        best_ask = None

        if bids:
            best_bid = max(
                (float(b.get("price", b.price)) if isinstance(b, dict) else float(b.price))
                for b in bids
            ) if bids else None

        if asks:
            best_ask = min(
                (float(a.get("price", a.price)) if isinstance(a, dict) else float(a.price))
                for a in asks
            ) if asks else None

        spread = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        return {"best_bid": best_bid, "best_ask": best_ask, "spread": spread}

    def get_fee_rate(self, token_id: str) -> int:
        """Fetch the fee rate for a token in basis points.

        Uses the CLOB /fee-rate endpoint. Returns 0 for fee-free markets.
        """
        self._throttle()
        try:
            import httpx
            resp = httpx.get(
                f"{self.host}/fee-rate",
                params={"token_id": token_id},
                timeout=10,
            )
            if resp.status_code == 200:
                return int(resp.json().get("fee_rate_bps", 0))
            return 0
        except Exception:
            logger.debug("Fee rate fetch failed for %s", token_id[:12])
            return 0

    # ── Integration with AvellanedaStoikov engine ────────────────────

    def quotes_from_mm_engine(
        self,
        bid_quote,
        ask_quote,
        token_id_yes: str,
        token_id_no: str = "",
    ) -> list[str]:
        """Post quotes produced by the AvellanedaStoikov engine.

        Accepts the (bid_quote, ask_quote) tuple returned by
        AvellanedaStoikovEngine.compute_quotes() and posts them via the CLOB.

        Args:
            bid_quote: ml.strategies.market_making.Quote or None.
            ask_quote: ml.strategies.market_making.Quote or None.
            token_id_yes: YES token ID for the market.
            token_id_no: NO token ID (reserved for future use).

        Returns:
            List of order IDs for successfully posted quotes.
        """
        order_ids: list[str] = []

        if bid_quote is not None:
            oid = self.post_limit_order(
                token_id=token_id_yes,
                side="BUY",
                price=bid_quote.price,
                size=bid_quote.size,
                order_type=getattr(bid_quote, "order_type", "GTC"),
            )
            if oid:
                order_ids.append(oid)

        if ask_quote is not None:
            oid = self.post_limit_order(
                token_id=token_id_yes,
                side="SELL",
                price=ask_quote.price,
                size=ask_quote.size,
                order_type=getattr(ask_quote, "order_type", "GTC"),
            )
            if oid:
                order_ids.append(oid)

        return order_ids
