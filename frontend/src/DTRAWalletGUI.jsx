
"""
PrivateDataCurrency_DAG.py — Wallet + Merchant + Data Import (Safe Demo + Tests)

Self-contained prototype of a private, non-USD, capped-supply retail token (DTRA) with a
Hashgraph-flavored DAG, privacy-preserving escrow, a CLI wallet with merchant mode,
AND Amazon/eBay shopping data import → convertible to DTRA at 100 USD = 1 DTRA.

This build AUTO-RUNS a non-interactive demo when stdin is not a TTY (e.g., sandbox).
Use --repl to force interactive mode if you have a real terminal.
Use --test to run the unit tests.

NOT production software. No legal/financial advice.
"""

from __future__ import annotations
import argparse
import base64
import csv
import dataclasses
import hashlib
import hmac
import json
import os
import secrets
import shlex
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple

# ========= Utility ========= #

def now_ts() -> float:
    return time.time()


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def to_atomic(amount_str: str, decimals: int = 8) -> int:
    s = str(amount_str).strip()
    if not s:
        raise ValueError("empty amount")
    neg = s.startswith("-")
    if s[0] in "+-":
        s = s[1:]
    if "." in s:
        whole, frac = s.split(".", 1)
    else:
        whole, frac = s, ""
    whole = "0" if whole == "" else whole
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ValueError("invalid numeric amount")
    frac = (frac + "0" * decimals)[:decimals]
    n = int(whole) * 10**decimals + (int(frac) if frac else 0)
    return -n if neg else n


def from_atomic(n: int, decimals: int = 8) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    whole = n // 10**decimals
    frac = n % 10**decimals
    return f"{sign}{whole}.{str(frac).zfill(decimals)}"

# ========= Lightweight signing ========= #

class Signer:
    def __init__(self, sk: bytes, vk: bytes, scheme: str):
        self.sk = sk
        self.vk = vk
        self.scheme = scheme

    @staticmethod
    def generate() -> "Signer":
        try:
            from nacl.signing import SigningKey # type: ignore
            sk_obj = SigningKey.generate()
            vk_obj = sk_obj.verify_key
            return Signer(sk_obj.encode(), vk_obj.encode(), scheme="ed25519")
        except Exception:
            # Fallback: HMAC scheme if ed25519 isn't available
            sk = secrets.token_bytes(32)
            vk = sha256(sk)
            return Signer(sk, vk, scheme="hmac")

    def address(self) -> str:
        return sha256(self.vk)[:20].hex()

    def sign(self, msg: bytes) -> str:
        if self.scheme == "ed25519":
            from nacl.signing import SigningKey # type: ignore
            sig = SigningKey(self.sk).sign(msg).signature
            return b64(sig)
        else:
            sig = hmac.new(self.sk, msg, hashlib.sha256).digest()
            return b64(sig)

# ========= Token & Transactions ========= #

TOKEN_SYMBOL = "DTRA"
TOKEN_DECIMALS = 8
TOKEN_CAP = 21_000_000 * 10**TOKEN_DECIMALS


def rand_id(prefix: str = "") -> str:
    return prefix + secrets.token_hex(16)


@dataclasses.dataclass
class TxBase:
    kind: str
    sender: str
    nonce: int
    payload: Dict[str, Any]
    ts: float
    pubkey_b64: str
    sig_b64: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "sender": self.sender,
            "nonce": self.nonce,
            "payload": self.payload,
            "ts": self.ts,
            "pubkey_b64": self.pubkey_b64,
            "sig_b64": self.sig_b64,
        }

    def id(self) -> str:
        return hashlib.sha256(canonical(self.to_dict())).hexdigest()


# ----- Transaction constructors ----- #

def build_tx(signer: Signer, kind: str, nonce: int, payload: Dict[str, Any]) -> TxBase:
    body = {
        "kind": kind,
        "sender": signer.address(),
        "nonce": nonce,
        "payload": payload,
        "ts": now_ts(),
        "pubkey_b64": b64(signer.vk),
    }
    sig = signer.sign(canonical(body))
    return TxBase(kind=kind, sender=body["sender"], nonce=nonce, payload=payload, ts=body["ts"],
                  pubkey_b64=body["pubkey_b64"], sig_b64=sig)


def tx_mint(signer: Signer, nonce: int, to_addr: str, amount: int) -> TxBase:
    return build_tx(signer, "mint", nonce, {"to": to_addr, "amount": amount})


def tx_transfer(signer: Signer, nonce: int, to_addr: str, amount: int, memo: str = "") -> TxBase:
    return build_tx(signer, "transfer", nonce, {"to": to_addr, "amount": amount, "memo": memo})

# Goods escrow

def tx_offer_goods(signer: Signer, nonce: int, goods_id: str, title: str, price: int, expiry_ts: float) -> TxBase:
    return build_tx(signer, "offer_goods", nonce, {
        "goods_id": goods_id,
        "title": title,
        "price": price,
        "expiry_ts": expiry_ts,
    })


def tx_accept_goods(signer: Signer, nonce: int, offer_tx_id: str, escrow_fund_tx_id: str) -> TxBase:
    return build_tx(signer, "accept_goods", nonce, {
        "offer_tx_id": offer_tx_id,
        "escrow_fund_tx_id": escrow_fund_tx_id,
    })


def tx_settle_goods(signer: Signer, nonce: int, offer_tx_id: str) -> TxBase:
    return build_tx(signer, "settle_goods", nonce, {"offer_tx_id": offer_tx_id})

# (Optional) Data trade (commit-reveal) — kept for completeness

def tx_offer_data(signer: Signer, nonce: int, data_commitment: str, key_commitment: str, price: int, expiry_ts: float) -> TxBase:
    return build_tx(signer, "offer_data", nonce, {
        "data_commitment": data_commitment,
        "key_commitment": key_commitment,
        "price": price,
        "expiry_ts": expiry_ts,
    })


def tx_accept_data(signer: Signer, nonce: int, offer_tx_id: str, escrow_fund_tx_id: str) -> TxBase:
    return build_tx(signer, "accept_data", nonce, {
        "offer_tx_id": offer_tx_id,
        "escrow_fund_tx_id": escrow_fund_tx_id,
    })


def tx_settle_data(signer: Signer, nonce: int, offer_tx_id: str, data_key_reveal_b64: str) -> TxBase:
    return build_tx(signer, "settle_data", nonce, {
        "offer_tx_id": offer_tx_id,
        "data_key_reveal_b64": data_key_reveal_b64,
    })


# ========= Events & DAG ========= #

@dataclasses.dataclass
class Event:
    creator: str
    parent_ids: List[str]
    tx_ids: List[str]
    ts: float
    id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"creator": self.creator, "parent_ids": self.parent_ids, "tx_ids": self.tx_ids, "ts": self.ts, "id": self.id}

    @staticmethod
    def create(creator: str, parent_ids: List[str], tx_ids: List[str]) -> "Event":
        body = {"creator": creator, "parent_ids": parent_ids, "tx_ids": tx_ids, "ts": now_ts()}
        eid = hashlib.sha256(canonical(body)).hexdigest()
        return Event(creator=creator, parent_ids=parent_ids, tx_ids=tx_ids, ts=body["ts"], id=eid)


# ========= Ledger State ========= #

class Ledger:
    def __init__(self):
        self.balances: Dict[str, int] = {}
        self.nonces: Dict[str, int] = {}
        self.total_supply: int = 0
        self.processed: Set[str] = set()
        # Offers / escrows: offer_id -> dict
        self.offers: Dict[str, Dict[str, Any]] = {}
        # Escrow balances live in balances under the escrow address key

    # ---- Helpers ---- #
    def ensure_acct(self, addr: str):
        if addr not in self.balances:
            self.balances[addr] = 0
        if addr not in self.nonces:
            self.nonces[addr] = 0

    def can_spend(self, addr: str, amount: int) -> bool:
        self.ensure_acct(addr)
        return self.balances[addr] >= amount

    # ---- Core rules ---- #
    def apply_event(self, txs: Dict[str, TxBase], ev: Event):
        for txid in ev.tx_ids:
            if txid in self.processed:
                continue
            tx = txs.get(txid)
            if not tx:
                continue
            self.apply_tx(tx)
            self.processed.add(txid)

    def apply_tx(self, tx: TxBase):
        kind = tx.kind
        self.ensure_acct(tx.sender)
        # Nonce check
        if tx.nonce != self.nonces[tx.sender]:
            return

        if kind == "mint":
            amt = int(tx.payload.get("amount", 0))
            to_addr = tx.payload.get("to")
            if to_addr and amt > 0 and self.total_supply + amt <= TOKEN_CAP:
                self.ensure_acct(to_addr)
                self.balances[to_addr] += amt
                self.total_supply += amt
                self.nonces[tx.sender] += 1
            return

        if kind == "transfer":
            to_addr = tx.payload.get("to")
            amt = int(tx.payload.get("amount", 0))
            if to_addr and amt > 0 and self.can_spend(tx.sender, amt):
                self.ensure_acct(to_addr)
                self.balances[tx.sender] -= amt
                self.balances[to_addr] += amt
                self.nonces[tx.sender] += 1
            return

        if kind == "offer_goods":
            price = int(tx.payload.get("price", 0))
            goods_id = tx.payload.get("goods_id")
            title = tx.payload.get("title", "")
            exp = float(tx.payload.get("expiry_ts", 0))
            if price <= 0 or not goods_id:
                return
            offer_id = tx.id()
            escrow_addr = self.derive_escrow_addr(offer_id)
            self.offers[offer_id] = {
                "type": "goods",
                "seller": tx.sender,
                "buyer": None,
                "price": price,
                "goods_id": goods_id,
                "title": title,
                "expiry_ts": exp,
                "escrow_addr": escrow_addr,
                "funded": False,
                "settled": False,
            }
            self.ensure_acct(escrow_addr)
            self.nonces[tx.sender] += 1
            return

        if kind == "accept_goods":
            offer_tx_id = tx.payload.get("offer_tx_id")
            fund_tx_id = tx.payload.get("escrow_fund_tx_id") # not enforced beyond presence
            offer = self.offers.get(offer_tx_id)
            if not offer or offer.get("type") != "goods":
                return
            if offer["expiry_ts"] and now_ts() > offer["expiry_ts"]:
                return
            escrow_addr = offer["escrow_addr"]
            price = offer["price"]
            if self.balances.get(escrow_addr, 0) >= price:
                offer["buyer"] = tx.sender
                offer["funded"] = True
                offer["fund_tx_id"] = fund_tx_id
                self.nonces[tx.sender] += 1
            return

        if kind == "settle_goods":
            offer_tx_id = tx.payload.get("offer_tx_id")
            offer = self.offers.get(offer_tx_id)
            if not offer or offer.get("type") != "goods":
                return
            if offer["seller"] != tx.sender or not offer.get("funded") or offer.get("settled"):
                return
            escrow_addr = offer["escrow_addr"]
            price = offer["price"]
            if self.balances.get(escrow_addr, 0) >= price:
                self.balances[escrow_addr] -= price
                self.balances[offer["seller"]] += price
                offer["settled"] = True
                self.nonces[tx.sender] += 1
            return

        # Optional data trades (commit‑reveal)
        if kind == "offer_data":
            price = int(tx.payload.get("price", 0))
            exp = float(tx.payload.get("expiry_ts", 0))
            data_cmt = tx.payload.get("data_commitment")
            key_cmt = tx.payload.get("key_commitment")
            if price <= 0 or not data_cmt or not key_cmt:
                return
            offer_id = tx.id()
            escrow_addr = self.derive_escrow_addr(offer_id)
            self.offers[offer_id] = {
                "type": "data",
                "seller": tx.sender,
                "buyer": None,
                "price": price,
                "data_commitment": data_cmt,
                "key_commitment": key_cmt,
                "expiry_ts": exp,
                "escrow_addr": escrow_addr,
                "funded": False,
                "settled": False,
            }
            self.ensure_acct(escrow_addr)
            self.nonces[tx.sender] += 1
            return

        if kind == "accept_data":
            offer_tx_id = tx.payload.get("offer_tx_id")
            offer = self.offers.get(offer_tx_id)
            if not offer or offer.get("type") != "data":
                return
            if offer["expiry_ts"] and now_ts() > offer["expiry_ts"]:
                return
            escrow_addr = offer["escrow_addr"]
            price = offer["price"]
            if self.balances.get(escrow_addr, 0) >= price:
                offer["buyer"] = tx.sender
                offer["funded"] = True
                self.nonces[tx.sender] += 1
            return

        if kind == "settle_data":
            offer_tx_id = tx.payload.get("offer_tx_id")
            reveal_b64 = tx.payload.get("data_key_reveal_b64")
            offer = self.offers.get(offer_tx_id)
            if not offer or offer.get("type") != "data" or offer["seller"] != tx.sender:
                return
            if not offer.get("funded") or offer.get("settled"):
                return
            try:
                key_bytes = base64.b64decode(reveal_b64)
            except Exception:
                return
            key_cmt = offer.get("key_commitment")
            if hashlib.sha256(key_bytes).hexdigest() != key_cmt:
                return
            escrow_addr = offer["escrow_addr"]
            price = offer["price"]
            if self.balances.get(escrow_addr, 0) >= price:
                self.balances[escrow_addr] -= price
                self.balances[offer["seller"]] += price
                offer["settled"] = True
                offer["reveal_b64"] = reveal_b64
                self.nonces[tx.sender] += 1
            return

    @staticmethod
    def derive_escrow_addr(offer_id: str) -> str:
        return sha256(("ESCROW:" + offer_id).encode()).hex()[:40]


# ========= Node (DAG + Gossip + Virtual Voting Finality) ========= #

class Node:
    def __init__(self, name: str, signer: Optional[Signer] = None):
        self.name = name
        self.signer = signer or Signer.generate()
        self.addr = self.signer.address()
        self.ledger = Ledger()
        self.txs: Dict[str, TxBase] = {}
        self.events: Dict[str, Event] = {}
        self.heads: Set[str] = set()
        self.finalized: Set[str] = set()
        self.seen_by: Dict[str, Set[str]] = {}
        self.peers: List["Node"] = []
        # Genesis event
        g = Event.create(creator=self.addr, parent_ids=[], tx_ids=[])
        self.events[g.id] = g
        self.heads.add(g.id)

    def connect(self, other: "Node"):
        if other is self:
            return
        if other not in self.peers:
            self.peers.append(other)
        if self not in other.peers:
            other.peers.append(self)

    def next_nonce(self) -> int:
        return self.ledger.nonces.get(self.addr, 0)

    def create_tx(self, kind: str, payload: Dict[str, Any]) -> TxBase:
        return build_tx(self.signer, kind, self.next_nonce(), payload)

    def submit_tx(self, tx: TxBase) -> str:
        txid = tx.id()
        self.txs[txid] = tx
        parent_ids = self.pick_parents()
        ev = Event.create(creator=self.addr, parent_ids=parent_ids, tx_ids=[txid])
        self.events[ev.id] = ev
        for p in parent_ids:
            self.heads.discard(p)
            self.seen_by.setdefault(p, set()).add(self.addr)
        self.heads.add(ev.id)
        return ev.id

    def pick_parents(self, k: int = 2) -> List[str]:
        if not self.heads:
            return []
        heads = list(self.heads)
        secrets.SystemRandom().shuffle(heads)
        return heads[:k]

    def receive_event(self, ev: Event):
        self.events[ev.id] = ev
        for p in ev.parent_ids:
            self.heads.discard(p)
            self.seen_by.setdefault(p, set()).add(self.addr)
        self.heads.add(ev.id)

    def receive_tx(self, txid: str, tx: TxBase):
        self.txs[txid] = tx

    def gossip_once(self):
        for peer in self.peers:
            # send unknown txs
            for txid, tx in self.txs.items():
                if txid not in peer.txs:
                    peer.receive_tx(txid, tx)
            # send unknown events
            for eid, ev in self.events.items():
                if eid not in peer.events:
                    peer.receive_event(ev)
            # receive from peer
            for txid, tx in peer.txs.items():
                if txid not in self.txs:
                    self.receive_tx(txid, tx)
            for eid, ev in peer.events.items():
                if eid not in self.events:
                    self.receive_event(ev)

    def virtual_vote_finalize(self, quorum_ratio: float = 0.67):
        all_nodes = len(self.peers) + 1
        threshold = max(1, int(all_nodes * quorum_ratio))
        # finalize any event seen by threshold nodes
        for eid, ev in list(self.events.items()):
            if eid in self.finalized:
                continue
            seen = len(self.seen_by.get(eid, set()))
            if ev.creator == self.addr:
                seen += 1
            if seen >= threshold:
                self.finalized.add(eid)
                self.apply_event_topo(ev)

    def apply_event_topo(self, ev: Event):
        stack: List[str] = []
        visited: Set[str] = set()

        def dfs(x: str):
            if x in visited:
                return
            visited.add(x)
            e = self.events.get(x)
            if not e:
                return
            for p in e.parent_ids:
                dfs(p)
            stack.append(x)

        dfs(ev.id)
        for eid in stack:
            e = self.events[eid]
            self.ledger.apply_event(self.txs, e)


# ========= Privacy‑Preserving Data Trade Helpers (optional) ========= #

class DataTrade:
    @staticmethod
    def commitment_of_data(data_bytes: bytes, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        salt = salt or secrets.token_bytes(16)
        c = hashlib.sha256(hashlib.sha256(data_bytes).digest() + salt).hexdigest()
        return c, salt

    @staticmethod
    def commitment_of_key(key_bytes: bytes) -> str:
        return hashlib.sha256(key_bytes).hexdigest()

    @staticmethod
    def gen_symmetric_key() -> bytes:
        return secrets.token_bytes(32)


# ========= Data Importers (Amazon / eBay CSV) ========= #

class DataImporter:
    """Parse simple exports of Amazon and eBay purchase histories.
    We keep this tolerant: it scans header names and tries common columns.
    Required: file path to CSV, returns dict with platform, orders count, total_usd.
    """

    @staticmethod
    def _parse_csv_sum_usd(path: str, amount_keys: List[str]) -> Tuple[int, float]:
        orders = 0
        total = 0.0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # find first present amount key
                amt = None
                for k in amount_keys:
                    if k in row and str(row[k]).strip():
                        amt = row[k]
                        break
                if amt is None:
                    continue
                # clean currency like "$12.34" or "USD 12.34"
                s = str(amt)
                s = s.replace("USD", "").replace("$", "").replace(",", "").strip()
                try:
                    val = float(s)
                except Exception:
                    continue
                total += max(val, 0.0)
                orders += 1
        return orders, total

    @staticmethod
    def import_amazon_csv(path: str) -> Dict[str, Any]:
        # Amazon order history often includes columns like: "Item Total", "Subtotal", "Total Charged"
        orders, total = DataImporter._parse_csv_sum_usd(path, [
            "Total Charged", "Item Total", "Subtotal", "Item Subtotal", "Grand Total",
        ])
        return {"platform": "amazon", "orders": orders, "total_usd": total}

    @staticmethod
    def import_ebay_csv(path: str) -> Dict[str, Any]:
        # eBay purchase history often includes: "Total Price", "Amount", "Total"
        orders, total = DataImporter._parse_csv_sum_usd(path, [
            "Total Price", "Amount", "Total", "Item total",
        ])
        return {"platform": "ebay", "orders": orders, "total_usd": total}


# ========= CLI Wallet App (with Merchant Mode + Data Import) ========= #

DEFAULT_USD_PER_DTRA = 100.0 # default conversion rate: $100 => 1 DTRA

class WalletApp:
    def __init__(self):
        # Three default participants
        self.nodes: Dict[str, Node] = {
            "merchant": Node("merchant"),
            "me": Node("me"),
            "data_minter": Node("data_minter"), # issuer for data-based mints
        }
        # connect peers all-to-all
        self.nodes["merchant"].connect(self.nodes["me"])
        self.nodes["merchant"].connect(self.nodes["data_minter"])
        self.nodes["me"].connect(self.nodes["data_minter"])
        # simple off-ledger merchant catalog: goods_id -> {title, price}
        self.catalog: Dict[str, Dict[str, Any]] = {}
        self.good_seq = 0
        self.current = "me"

    # ---- Helpers ---- #
    def node(self) -> Node:
        return self.nodes[self.current]

    def other(self, name: str) -> Node:
        return self.nodes[name]

    def sync(self, rounds: int = 3):
        for _ in range(rounds):
            for n in self.nodes.values():
                n.gossip_once()
            for n in self.nodes.values():
                n.virtual_vote_finalize()

    def await_offer_visible(self, owner: Node, offer_id: str, attempts: int = 12) -> bool:
        """Wait for an offer to be finalized & applied on the owner's ledger.
        Fixes KeyError when the offer isn't yet in ledger.offers after submit_tx()."""
        for _ in range(max(1, attempts)):
            if offer_id in owner.ledger.offers:
                return True
            self.sync(1)
        return False

    def fmt_amt(self, a: int) -> str:
        return f"{from_atomic(a)} {TOKEN_SYMBOL}"

    def print_balances(self):
        for label, n in self.nodes.items():
            bal = n.ledger.balances.get(n.addr, 0)
            print(f"{label:12s} {n.addr[:10]}… {self.fmt_amt(bal)}")

    def _usd_to_dtra_atomic(self, usd: float, usd_per_dtra: float) -> int:
        dtra = usd / max(usd_per_dtra, 1e-9)
        return to_atomic(f"{dtra:.8f}")

    def _mint_from_data(self, summary: Dict[str, Any], usd_per_dtra: float):
        usd = summary.get("total_usd", 0.0)
        dtra_atomic = self._usd_to_dtra_atomic(usd, usd_per_dtra)
        if dtra_atomic <= 0:
            print("no value detected; nothing to mint")
            return
        # use data_minter as the issuing authority
        minter = self.other("data_minter")
        recipient = self.node()
        tx = tx_mint(minter.signer, minter.next_nonce(), recipient.addr, dtra_atomic)
        minter.submit_tx(tx)
        self.sync()
        print(f"minted {self.fmt_amt(dtra_atomic)} to {recipient.addr[:10]}… based on {summary['platform']} data (${usd:.2f}) @ ${usd_per_dtra:.2f}/DTRA")

    # ---- Commands ---- #
    def cmd_help(self):
        print(
            """
Commands:
  help Show this help
  whoami Show current profile & address
  use <me|merchant|data_minter> Switch active wallet
  address Show current address
  balance Show balances of all known wallets
  peers List peer addresses

  mint <amount> Demo mint to current wallet (DTRA units)
  transfer <to_addr> <amount> Transfer DTRA to address
  gossip Run a gossip+finality round

  merchant_add "Title" <price> (merchant) Add a catalog item priced in DTRA
  merchant_goods View merchant catalog
  merchant_checkout <goods_id> (buyer) Create offer, fund escrow, accept
  offers List on-ledger offers (goods/data)
  settle <offer_id> (merchant) Release escrow to merchant

  data_value_amazon <csv> Parse Amazon CSV → show USD total + DTRA @ $100/DTRA
  data_value_ebay <csv> Parse eBay CSV → show USD total + DTRA @ $100/DTRA
  data_mint_amazon <csv> [usd_per_dtra] Mint DTRA based on Amazon CSV (default $100/DTRA)
  data_mint_ebay <csv> [usd_per_dtra] Mint DTRA based on eBay CSV (default $100/DTRA)

Examples:
  mint 100
  merchant_add "Coffee" 1.25
  merchant_goods
  merchant_checkout g_1
  data_value_amazon ./amazon_orders.csv
  data_mint_ebay ./ebay_history.csv 100
            """
        )

    def cmd_whoami(self):
        n = self.node()
        print(f"current: {self.current}\naddress: {n.addr}")

    def cmd_use(self, args: List[str]):
        if not args:
            print("usage: use <me|merchant|data_minter>")
            return
        who = args[0].strip().lower()
        if who not in self.nodes:
            print("unknown profile", who)
            return
        self.current = who
        print(f"switched to {who} :: {self.node().addr}")

    def cmd_address(self):
        print(self.node().addr)

    def cmd_balance(self):
        self.sync(2)
        self.print_balances()

    def cmd_peers(self):
        n = self.node()
        if not n.peers:
            print("(no peers)")
            return
        for p in n.peers:
            print(f"{p.name:12s} {p.addr}")

    def cmd_gossip(self):
        self.sync(2)
        print("gossip+finalize done")

    def cmd_mint(self, args: List[str]):
        if not args:
            print("usage: mint <amount>")
            return
        try:
            amt = to_atomic(args[0])
        except Exception as e:
            print("invalid amount:", e)
            return
        n = self.node()
        tx = tx_mint(n.signer, n.next_nonce(), n.addr, amt)
        n.submit_tx(tx)
        self.sync()
        print(f"minted {self.fmt_amt(amt)} to {n.addr[:10]}…")

    def cmd_transfer(self, args: List[str]):
        if len(args) < 2:
            print("usage: transfer <to_addr> <amount>")
            return
        to = args[0]
        try:
            amt = to_atomic(args[1])
        except Exception as e:
            print("invalid amount:", e)
            return
        n = self.node()
        tx = tx_transfer(n.signer, n.next_nonce(), to, amt)
        n.submit_tx(tx)
        self.sync()
        print(f"sent {self.fmt_amt(amt)} → {to[:10]}…")

    # ---- Merchant features ---- #
    def cmd_merchant_add(self, args: List[str]):
        if self.current != "merchant":
            print("switch to merchant first: use merchant")
            return
        if len(args) < 2:
            print('usage: merchant_add "Title" <price>')
            return
        # Title parsing: quoted or single token
        if args[0].startswith(('"', "'")) and not args[0].endswith(('"', "'")):
            tokens = [args[0]]
            i = 1
            while i < len(args) and not args[i].endswith(('"', "'")):
                tokens.append(args[i]); i += 1
            if i < len(args):
                tokens.append(args[i]); i += 1
            title = " ".join(tokens)[1:-1]
            rest = args[i:]
        else:
            title = args[0].strip('\"\'')
            rest = args[1:]
        if not rest:
            print('usage: merchant_add "Title" <price>')
            return
        try:
            price = to_atomic(rest[0])
        except Exception as e:
            print("invalid price:", e)
            return
        self.good_seq += 1
        gid = f"g_{self.good_seq}"
        self.catalog[gid] = {"title": title, "price": price}
        print(f"added {gid}: '{title}' @ {self.fmt_amt(price)}")

    def cmd_merchant_goods(self):
        if not self.catalog:
            print("(merchant catalog empty)")
            return
        print("goods_id price title")
        print("-------------------------------------------")
        for gid, rec in sorted(self.catalog.items()):
            print(f"{gid:7s} {from_atomic(rec['price']):>12s} {TOKEN_SYMBOL} {rec['title']}")

    def cmd_merchant_checkout(self, args: List[str]):
        if not args:
            print("usage: merchant_checkout <goods_id>")
            return
        gid = args[0]
        if gid not in self.catalog:
            print("unknown goods_id", gid)
            return
        buyer = self.node()
        merch = self.other("merchant")
        price = self.catalog[gid]["price"]
        title = self.catalog[gid]["title"]
        # 1) Merchant posts an on-ledger offer
        offer_tx = tx_offer_goods(merch.signer, merch.next_nonce(), gid, title, price, now_ts() + 3600)
        merch.submit_tx(offer_tx)
        self.sync()
        offer_id = offer_tx.id()
        if not self.await_offer_visible(merch, offer_id, attempts=16):
            print("error: offer not yet finalized; try 'gossip' and retry")
            return
        escrow_addr = merch.ledger.offers[offer_id]["escrow_addr"]
        # 2) Buyer funds escrow via transfer
        fund_tx = tx_transfer(buyer.signer, buyer.next_nonce(), escrow_addr, price, memo=f"escrow {gid}")
        buyer.submit_tx(fund_tx)
        self.sync()
        # 3) Buyer accepts, referencing the funding tx (for traceability)
        accept_tx = tx_accept_goods(buyer.signer, buyer.next_nonce(), offer_id, fund_tx.id())
        buyer.submit_tx(accept_tx)
        self.sync()
        print("checkout created")
        print(" offer_id :", offer_id)
        print(" goods :", gid, title)
        print(" escrow_addr:", escrow_addr)
        print(" price :", self.fmt_amt(price))
        print(" status : funded=", merch.ledger.offers[offer_id]["funded"], " settled=", merch.ledger.offers[offer_id]["settled"])
        print(" next step : merchant runs → settle", offer_id)

    def cmd_offers(self):
        # Show offers from current node's view
        n = self.node()
        if not n.ledger.offers:
            print("(no offers on ledger)")
            return
        hdr = "offer_id type price funded settled seller.. buyer.. title/goods"
        print(hdr)
        print("-" * len(hdr))
        for oid, o in n.ledger.offers.items():
            price = from_atomic(o.get("price", 0))
            t = o.get("type")
            s = o.get("seller", "")[:10] + "…"
            b = (o.get("buyer", "") or "")[:10] + ("…" if o.get("buyer") else "")
            funded = "Y" if o.get("funded") else "N"
            settled = "Y" if o.get("settled") else "N"
            title = o.get("title") or o.get("goods_id") or (o.get("data_commitment", "")[:10] + "…")
            print(f"{oid[:8]} {t:5s} {price:>12s} {TOKEN_SYMBOL} {funded:^6s} {settled:^7s} {s:10s} {b:10s} {title}")

    def cmd_settle(self, args: List[str]):
        if self.current != "merchant":
            print("settle must be run by merchant: use merchant")
            return
        if not args:
            print("usage: settle <offer_id>")
            return
        oid = args[0]
        merch = self.node()
        if oid not in merch.ledger.offers:
            print("unknown offer_id")
            return
        tx = tx_settle_goods(merch.signer, merch.next_nonce(), oid)
        merch.submit_tx(tx)
        self.sync()
        offer = merch.ledger.offers.get(oid, {})
        print("settled:", oid, " funded=", offer.get("funded"), " settled=", offer.get("settled"))

    # ---- Data import commands ---- #
    def cmd_data_value_amazon(self, args: List[str]):
        if not args:
            print("usage: data_value_amazon <csv>")
            return
        path = args[0]
        info = DataImporter.import_amazon_csv(path)
        est = self._usd_to_dtra_atomic(info["total_usd"], DEFAULT_USD_PER_DTRA)
        print(f"AMAZON: orders={info['orders']} total=${info['total_usd']:.2f} → ~{from_atomic(est)} {TOKEN_SYMBOL} @ ${DEFAULT_USD_PER_DTRA}/DTRA")

    def cmd_data_value_ebay(self, args: List[str]):
        if not args:
            print("usage: data_value_ebay <csv>")
            return
        path = args[0]
        info = DataImporter.import_ebay_csv(path)
        est = self._usd_to_dtra_atomic(info["total_usd"], DEFAULT_USD_PER_DTRA)
        print(f"EBAY: orders={info['orders']} total=${info['total_usd']:.2f} → ~{from_atomic(est)} {TOKEN_SYMBOL} @ ${DEFAULT_USD_PER_DTRA}/DTRA")

    def cmd_data_mint_amazon(self, args: List[str]):
        if not args:
            print("usage: data_mint_amazon <csv> [usd_per_dtra]")
            return
        path = args[0]
        rate = float(args[1]) if len(args) > 1 else DEFAULT_USD_PER_DTRA
        info = DataImporter.import_amazon_csv(path)
        self._mint_from_data(info, rate)

    def cmd_data_mint_ebay(self, args: List[str]):
        if not args:
            print("usage: data_mint_ebay <csv> [usd_per_dtra]")
            return
        path = args[0]
        rate = float(args[1]) if len(args) > 1 else DEFAULT_USD_PER_DTRA
        info = DataImporter.import_ebay_csv(path)
        self._mint_from_data(info, rate)

    # ---- REPL ---- #
    def repl(self):
        print("DTRA Wallet + Merchant + Data Import (no USD) ready. Type 'help' for commands.")
        try:
            while True:
                prompt = f"(dtra)[{self.current}]$ "
                try:
                    line = input(prompt)
                except (EOFError, OSError):
                    print("\n(stdin not available — exiting REPL)\n")
                    break
                except KeyboardInterrupt:
                    print()
                    break
                if not line.strip():
                    continue
                try:
                    parts = shlex.split(line)
                except Exception as e:
                    print("parse error:", e)
                    continue
                cmd, *args = parts
                cmd = cmd.lower()
                try:
                    if cmd == "help":
                        self.cmd_help()
                    elif cmd == "whoami":
                        self.cmd_whoami()
                    elif cmd == "use":
                        self.cmd_use(args)
                    elif cmd == "address":
                        self.cmd_address()
                    elif cmd == "balance":
                        self.cmd_balance()
                    elif cmd == "peers":
                        self.cmd_peers()
                    elif cmd == "gossip":
                        self.cmd_gossip()
                    elif cmd == "mint":
                        self.cmd_mint(args)
                    elif cmd == "transfer":
                        self.cmd_transfer(args)
                    elif cmd == "merchant_add":
                        self.cmd_merchant_add(args)
                    elif cmd == "merchant_goods":
                        self.cmd_merchant_goods()
                    elif cmd == "merchant_checkout":
                        self.cmd_merchant_checkout(args)
                    elif cmd == "offers":
                        self.cmd_offers()
                    elif cmd == "settle":
                        self.cmd_settle(args)
                    elif cmd == "data_value_amazon":
                        self.cmd_data_value_amazon(args)
                    elif cmd == "data_value_ebay":
                        self.cmd_data_value_ebay(args)
                    elif cmd == "data_mint_amazon":
                        self.cmd_data_mint_amazon(args)
                    elif cmd == "data_mint_ebay":
                        self.cmd_data_mint_ebay(args)
                    elif cmd in {"quit", "exit"}:
                        break
                    else:
                        print("unknown command; try 'help'")
                except Exception as e:
                    print("error:", e)
        finally:
            print("bye")


# ========= Automated Demo ========= #

def run_demo():
    print("[DTRA demo] starting…")
    app = WalletApp()
    # Mint some buyer funds
    app.current = "me"; app.cmd_mint(["5"]) # 5 DTRA
    # Merchant adds product and buyer purchases
    app.current = "merchant"; app.cmd_merchant_add(["Coffee", "1.50"]) # add product
    app.current = "me"; app.cmd_merchant_goods(); app.cmd_merchant_checkout(["g_1"]) # buy it
    # Merchant settles
    app.current = "merchant"; offers = list(app.nodes["merchant"].ledger.offers.keys())
    if offers: app.cmd_settle([offers[0]])
    # Show balances
    app.current = "me"; app.cmd_balance()
    print("[DTRA demo] done.")


# ========= Tests ========= #

class TestDTRA(unittest.TestCase):
    def test_to_from_atomic(self):
        self.assertEqual(to_atomic("1.50"), 150000000)
        self.assertEqual(to_atomic("0.00000001"), 1)
        self.assertEqual(from_atomic(150000000), "1.50000000")
        self.assertEqual(from_atomic(1), "0.00000001")

    def test_mint_and_transfer(self):
        app = WalletApp()
        me = app.nodes["me"]
        merch = app.nodes["merchant"]
        amt = to_atomic("10")
        me.submit_tx(tx_mint(me.signer, me.next_nonce(), me.addr, amt))
        app.sync()
        self.assertEqual(me.ledger.balances.get(me.addr, 0), amt)
        xfer = to_atomic("3")
        me.submit_tx(tx_transfer(me.signer, me.next_nonce(), merch.addr, xfer))
        app.sync()
        self.assertEqual(me.ledger.balances.get(me.addr, 0), amt - xfer)
        self.assertEqual(me.ledger.balances.get(merch.addr, 0), xfer)

    def test_data_import_conversion(self):
        # Create temp Amazon CSV
        headers = ["Order ID", "Item", "Total Charged"]
        rows = [
            ["1", "A", "$12.00"],
            ["2", "B", "$18.00"], # total = $30 => 0.30000000 DTRA @ $100/DTRA
        ]
        fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(headers); w.writerows(rows)
        try:
            info = DataImporter.import_amazon_csv(path)
            self.assertEqual(info["orders"], 2)
            self.assertAlmostEqual(info["total_usd"], 30.0, places=2)
            app = WalletApp(); app.current = "me"
            est = app._usd_to_dtra_atomic(info["total_usd"], DEFAULT_USD_PER_DTRA)
            self.assertEqual(from_atomic(est), "0.30000000")
        finally:
            os.remove(path)

        # Create temp eBay CSV
        headers = ["Txn", "Item", "Total Price"]
        rows = [
            ["x", "Y", "USD 12.50"],
            ["y", "Z", "$7.50"], # total = $20 => 0.20000000 DTRA
        ]
        fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(headers); w.writerows(rows)
        try:
            info = DataImporter.import_ebay_csv(path)
            self.assertEqual(info["orders"], 2)
            self.assertAlmostEqual(info["total_usd"], 20.0, places=2)
            app = WalletApp(); app.current = "me"
            est = app._usd_to_dtra_atomic(info["total_usd"], DEFAULT_USD_PER_DTRA)
            self.assertEqual(from_atomic(est), "0.20000000")
        finally:
            os.remove(path)

    def test_checkout_flow_finalization_wait(self):
        app = WalletApp()
        app.current = "me"; app.cmd_mint(["3"]) # Ensure buyer has funds
        app.current = "merchant"; app.cmd_merchant_add(["Tea", "1.00"]) # Add goods
        app.current = "me"; app.cmd_merchant_checkout(["g_1"]) # Triggers await_offer_visible
        merch = app.nodes["merchant"]
        # There should be at least one goods offer, funded
        self.assertTrue(any(o.get("type") == "goods" for o in merch.ledger.offers.values()))
        oid = next(iter(merch.ledger.offers.keys()))
        self.assertTrue(merch.ledger.offers[oid].get("funded", False))
        # Merchant can settle
        app.current = "merchant"; app.cmd_settle([oid])
        self.assertTrue(merch.ledger.offers[oid].get("settled", False))


# ========= Entry ========= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run automated demo")
    parser.add_argument("--repl", action="store_true", help="force REPL (requires real stdin)")
    parser.add_argument("--test", action="store_true", help="run unit tests and exit")
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0]], exit=False)
    elif args.demo or (not sys.stdin.isatty() and not args.repl):
        run_demo()
    else:
        try:
            WalletApp().repl()
        except OSError:
            print("[error] stdin not available, falling back to demo…")
            run_demo()	
