# SpaceProof

**Space-grade proof infrastructure. No receipt, not real.**

Part of the ProofChain series: SpaceProof | SpendProof | ClaimProof | VoteProof | OriginProof | GreenProof

## What SpaceProof Does

| Module | Purpose | Value |
|--------|---------|-------|
| compress | 10x telemetry compression | Bandwidth savings |
| witness | Physics law discovery (KAN/MDL) | Accelerated R&D |
| sovereignty | Autonomy threshold calculation | Mars crew sizing |
| detect | Entropy anomaly detection | Fraud detection |
| ledger | Receipt storage | Full audit trail |
| anchor | Merkle proofs | Tamper-proof |
| loop | 60-second SENSE->ACTUATE cycle | Automated improvement |

## Quick Start

```bash
# Install
pip install -e .

# Run tests
pytest tests/ -v

# CLI usage
python cli.py --help

# Test receipt emission
python cli.py --test

# Run with stakeholder config
python cli.py --config xai --test
python cli.py --config doge --test
```

## Stakeholder Configs

| Config | Stakeholder | Primary Modules | Key Value |
|--------|-------------|-----------------|-----------|
| `configs/xai.yaml` | Elon/xAI | compress, witness, sovereignty | Mars crew threshold |
| `configs/doge.yaml` | DOGE | ledger, detect, anchor | $162B fraud target |
| `configs/dot.yaml` | DOT | compress, ledger, detect | Compliance |
| `configs/defense.yaml` | Defense | compress, ledger, anchor | Fire-control lineage |
| `configs/nro.yaml` | NRO | compress, anchor, sovereignty | Constellation governance |

## Module Taxonomy

```
core.py (foundation)
    |
compress.py <- witness.py
    |           |
sovereignty.py <- detect.py
    |
  ledger.py <- anchor.py
    |
  loop.py (harness)
```

## Domain Generators

- `domains/galaxy.py` - Galaxy rotation curve generation
- `domains/colony.py` - Mars colony state simulation
- `domains/telemetry.py` - Fleet telemetry (Tesla/Starlink/SpaceX)

## SLOs

| Module | Metric | Threshold | Stoprule |
|--------|--------|-----------|----------|
| compress | compression_ratio | >=10 | FAIL |
| compress | recall | >=0.999 | FAIL |
| witness | training_time | <=60s | WARN |
| detect | false_positive_rate | <0.01 | FAIL |
| loop | cycle_time | <=60s | WARN |
| anchor | verify_time | <=2s | WARN |

## Core Primitives

All modules import from `spaceproof.core`:

```python
from spaceproof.core import dual_hash, emit_receipt, merkle, StopRule
```

- `dual_hash(data)` - SHA256:BLAKE3 format
- `emit_receipt(type, data)` - Create CLAUDEME-compliant receipt
- `merkle(items)` - Compute Merkle root
- `StopRule` - Exception for stoprule violations

## Gates

### T+2h: SKELETON
- [ ] spec.md exists
- [ ] ledger_schema.json exists
- [ ] cli.py emits receipt

### T+24h: MVP
- [ ] All modules importable
- [ ] All tests pass
- [ ] All receipts emit

### T+48h: HARDENED
- [ ] 80% coverage on core
- [ ] All configs work
- [ ] Documentation complete

## The Physics

Information theory unifies all domains:
- **Compression = Discovery** - High compression ratio reveals underlying structure
- **Entropy = Fraud Detection** - Anomalies in entropy signal violations
- **Bits/sec = Sovereignty** - When internal processing exceeds external input
- **Receipts = Trust** - Immutable audit trail for every operation

## ProofChain Series

SpaceProof is the flagship engine. Future repositories:
- **SpendProof** - Government spending (uses detect + ledger)
- **ClaimProof** - Healthcare fraud (uses detect + anchor)
- **VoteProof** - Election integrity (uses anchor + ledger)
- **OriginProof** - Supply chain (uses compress + anchor)
- **GreenProof** - Climate accountability (uses detect + witness)

## License

MIT

---

*Space-grade proof. No receipt, not real. Ship at T+24h or kill.*
