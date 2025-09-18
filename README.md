# The Nova AI POC

```bash
python3 scripts/infer.py > data/instance.json
cargo build --release
./target/release/prover data/instance.json out
./target/release/verifier out/vk.bin out/proof.bin out/public_inputs.json
```
