"""
audit.py
==================================================================
Append-only, tamper-evident audit trail for every inference. Each record is
hash-chained to the previous one (record_hash = sha256(prev_hash + payload)),
so any later edit or deletion breaks the chain and is detectable by ``verify``.

Records carry provenance, never PHI: a content hash of the *processed pixel
data* (not the file, not patient tags), the model + code version, whether
de-identification ran, and a summary of the output. This lets you answer
"what produced this result?" without storing anything identifying.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

GENESIS = "0" * 64


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(payload: Dict[str, Any]) -> bytes:
    """Deterministic JSON encoding for hashing (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class AuditLog:
    """
    JSONL audit log with a per-record hash chain.

    Usage
    -----
    >>> log = AuditLog("audit.jsonl")
    >>> log.record(model_key="ct_chest_nodule_seg", input_sha256=h,
    ...            deid_applied=True, top_finding="nodule", confidence=0.87)
    >>> ok, bad_line = log.verify()   # (True, None) when intact
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- internals ---------------------------------------------------------
    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)["record_hash"]
                    except Exception:
                        continue
        return last

    # -- write -------------------------------------------------------------
    def record(self, **fields: Any) -> Dict[str, Any]:
        """Append one audit record and return it (including its chain hashes)."""
        payload = {
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        prev = self._last_hash()
        record_hash = sha256_hex(prev.encode() + _canonical(payload))
        entry = {**payload, "prev_hash": prev, "record_hash": record_hash}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        return entry

    # -- read / verify -----------------------------------------------------
    def verify(self) -> Tuple[bool, Optional[int]]:
        """
        Re-walk the chain. Returns (True, None) if intact, else (False, line#)
        of the first record whose hash or link doesn't check out.
        """
        if not self.path.exists():
            return True, None
        prev = GENESIS
        with self.path.open() as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    return False, i
                payload = {k: v for k, v in entry.items()
                           if k not in ("prev_hash", "record_hash")}
                expect = sha256_hex(prev.encode() + _canonical(payload))
                if entry.get("prev_hash") != prev or entry.get("record_hash") != expect:
                    return False, i
                prev = entry["record_hash"]
        return True, None

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open() as fh:
            return sum(1 for line in fh if line.strip())


if __name__ == "__main__":  # pragma: no cover
    import tempfile
    log = AuditLog(Path(tempfile.mkdtemp()) / "audit.jsonl")
    for i in range(3):
        log.record(model_key="demo", input_sha256=sha256_hex(bytes([i])),
                   deid_applied=True, top_finding="x", confidence=0.5 + i / 10)
    print("records:", len(log), "| intact:", log.verify())
    # tamper: rewrite a line
    lines = log.path.read_text().splitlines()
    lines[1] = lines[1].replace('"demo"', '"tampered"')
    log.path.write_text("\n".join(lines) + "\n")
    print("after tamper:", log.verify())
