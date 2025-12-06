#!/bin/bash
# Quick test all new experiments
# Run from project root

echo "Testing new experiments (2 epochs each)..."
echo ""

echo "1. Testing 5s8a LSTM..."
python -m experiments.run_icl_5s8a --model lstm --epochs 2 --no-verbose
echo ""

echo "2. Testing 8s8a LSTM..."
python -m experiments.run_icl_8s8a --model lstm --epochs 2 --no-verbose
echo ""

echo "3. Testing absorption state..."
python -m experiments.run_icl_absorption --model lstm --epochs 2 --no-verbose
echo ""

echo "4. Testing deep RNN..."
python -m experiments.run_icl_deep_rnn --epochs 2 --no-verbose
echo ""

echo "5. Testing small LSTM (d_model=64)..."
python -m experiments.run_icl_small_models --model lstm --epochs 2 --no-verbose
echo ""

echo "6. Testing small vanilla RNN (d_model=64)..."
python -m experiments.run_icl_small_models --model vanilla_rnn --epochs 2 --no-verbose
echo ""

echo "7. Testing small deep RNN (d_model=64)..."
python -m experiments.run_icl_small_models --model deep_rnn --epochs 2 --no-verbose
echo ""

echo "All tests complete!"
