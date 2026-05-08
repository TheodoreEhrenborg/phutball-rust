#!/bin/bash
set -e
cd /gt/phutball/polecats/jasper/phutball

BIN=./target/release/phutball-rust
LOG=/tmp/nnue_v2_pipeline.log

exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] Starting NNUE v2 pipeline"

echo "[$(date)] Step 1: Generate 500 games with eval5:500ms teacher"
$BIN nnue-gen-data --games 500 --engine eval5:500 --out nnue_v2.dat
echo "[$(date)] Data generation complete"

echo "[$(date)] Step 2: Train 100 epochs"
$BIN nnue-train --data nnue_v2.dat --epochs 100 --save nnue.bin
echo "[$(date)] Training complete"

echo "[$(date)] Step 3: Tournament nnue-eval:1000 vs eval5:1000"
$BIN tournament nnue-eval:1000 eval5:1000 2>&1 | tee /tmp/nnue_vs_eval5.log
echo "[$(date)] Tournament 1 complete"

echo "[$(date)] Step 4: Tournament nnue-eval:1000 vs eval2:1000"
$BIN tournament nnue-eval:1000 eval2:1000 2>&1 | tee /tmp/nnue_vs_eval2.log
echo "[$(date)] Tournament 2 complete"

echo "[$(date)] PIPELINE COMPLETE"
