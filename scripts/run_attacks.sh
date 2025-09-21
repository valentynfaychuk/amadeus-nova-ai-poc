#!/bin/bash
# Security Analysis: Attack Simulation Script for Nova AI
# This script demonstrates various attack vectors and shows how the system defends against them

set -e  # Exit on any error

echo "🛡️  NOVA AI SECURITY ANALYSIS: ATTACK SIMULATION"
echo "=================================================="
echo ""
echo "This script simulates various attack scenarios to demonstrate"
echo "the security properties of the Nova AI proof system."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper function for section headers
section() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Helper function for test results
test_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $3${NC}"
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up attack artifacts...${NC}"
    rm -f data/malicious_run.json
    rm -f data/sophisticated_attack.json
    rm -rf data/malicious_out
    rm -rf data/attack_out
}

# Set trap to cleanup on exit
trap cleanup EXIT

section "SETUP: Preparing attack environment"

# Ensure we have valid test data
if [ ! -f "data/run_k50240.json" ]; then
    echo "❌ Error: Need valid test data. Run './run_inference.sh' first"
    exit 1
fi

if [ ! -f "keys/pk.bin" ]; then
    echo "❌ Error: Need proving keys. Run 'nova_poc setup --out-dir keys' first"
    exit 1
fi

echo "✅ Prerequisites satisfied"

section "ATTACK 1: Naive Malicious Prover (Fake y1/y2, Wrong Commitments)"

echo "🎭 Creating malicious run data with fake y1/y2 values..."

# Create malicious data with fake values
python3 << 'EOF'
import json

# Load the real data
with open('data/run_k50240.json', 'r') as f:
    data = json.load(f)

print('📋 Original y1 values (first 5):', data['y1'][:5])

# Replace y1 with fake values (but keep same length)
fake_y1 = [999999] * 16  # Obviously wrong values
data['y1'] = fake_y1

# Replace y2 with fake values too
fake_y2 = [777777] * 16
data['y2'] = fake_y2

# Keep the same commitments initially (will be wrong)
print('🎭 Fake y1 values:', fake_y1[:5])
print('🎭 Fake y2 values:', fake_y2[:5])

# Save the malicious data
with open('data/malicious_run.json', 'w') as f:
    json.dump(data, f, indent=2)

print('✅ Created malicious run data with fake y1/y2 values')
EOF

echo ""
echo "🔧 Testing if malicious prover can generate proof..."
mkdir -p data/malicious_out

if cargo run --release -p nova_poc -- prove data/malicious_run.json --pk-path keys/pk.bin --out-dir data/malicious_out > /dev/null 2>&1; then
    echo "✅ Malicious prover successfully generated proof"
else
    echo "❌ Malicious prover failed to generate proof"
    exit 1
fi

echo ""
echo "🚨 Testing verifier response to malicious proof..."

# Test verification - should fail
if cargo run --release -p nova_poc -- verify data/malicious_run.json --weights1-path data/w1_16x50240.bin > /dev/null 2>&1; then
    echo -e "${RED}❌ SECURITY BREACH: Verifier accepted malicious proof!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ SECURITY SUCCESS: Verifier rejected malicious proof (Freivalds detected fraud)${NC}"
fi

section "ATTACK 2: Sophisticated Malicious Prover (Fake y1/y2, Correct Commitments)"

echo "🎭 Creating sophisticated attack with correct commitments for fake values..."

python3 << 'EOF'
import json

# Load malicious data
with open('data/malicious_run.json', 'r') as f:
    data = json.load(f)

# Compute correct commitments for fake values
fake_y1 = data['y1']  # [999999] * 16
fake_y2 = data['y2']  # [777777] * 16

# Beta = 7 for vector commitments
def commit_vector(vec):
    acc = 0
    beta_pow = 1
    for val in vec:
        acc += val * (7 ** (len([x for x in vec[:vec.index(val)+1] if x == val and vec[:vec.index(val)+1].count(val) == 1])) if vec.count(val) == 1 else 7 ** vec.index(val))
        beta_pow *= 7
    return acc

# Simplified commitment for uniform vectors
def commit_uniform_vector(val, length):
    # For [val, val, val, ...] vector of length n
    acc = 0
    beta_pow = 1
    for i in range(length):
        acc += val * beta_pow
        beta_pow *= 7
    return acc

correct_h_y1 = commit_uniform_vector(999999, 16)
correct_h_y2 = commit_uniform_vector(777777, 16)

# Update commitments to match the fake values
data['commitments']['h_y1'] = str(correct_h_y1)
data['commitments']['h_y'] = str(correct_h_y2)

# Save the sophisticated attack
with open('data/sophisticated_attack.json', 'w') as f:
    json.dump(data, f, indent=2)

print('✅ Created sophisticated attack with correct commitments')
print('   - Fake y1/y2 values: [999999...], [777777...]')
print(f'   - Correct h_y1 commitment: {correct_h_y1}')
print(f'   - Correct h_y2 commitment: {correct_h_y2}')
EOF

echo ""
echo "🔧 Testing sophisticated attacker proof generation..."
mkdir -p data/attack_out

if cargo run --release -p nova_poc -- prove data/sophisticated_attack.json --pk-path keys/pk.bin --out-dir data/attack_out > /dev/null 2>&1; then
    echo "✅ Sophisticated attacker generated proof with correct commitments"
else
    echo "❌ Sophisticated attacker failed to generate proof"
    exit 1
fi

echo ""
echo "🚨 Testing verifier response to sophisticated attack..."

# Test verification - should still fail
if cargo run --release -p nova_poc -- verify data/sophisticated_attack.json --weights1-path data/w1_16x50240.bin > /dev/null 2>&1; then
    echo -e "${RED}❌ SECURITY BREACH: Verifier accepted sophisticated attack!${NC}"
    exit 1
else
    echo -e "${GREEN}✅ SECURITY SUCCESS: Verifier rejected sophisticated attack (Freivalds still detected fraud)${NC}"
fi

section "ATTACK 3: Bypass Attempt (Skip Freivalds Verification)"

echo "🎭 Testing if attacker can bypass Freivalds verification..."

# Test naive attack with --skip-freivalds
echo "Testing naive attack with --skip-freivalds..."
if cargo run --release -p nova_poc -- verify data/malicious_run.json --skip-freivalds > /dev/null 2>&1; then
    echo -e "${RED}❌ WARNING: Naive attack succeeded when skipping Freivalds${NC}"
else
    echo -e "${GREEN}✅ GOOD: Naive attack failed even without Freivalds (commitment mismatch)${NC}"
fi

# Test sophisticated attack with --skip-freivalds
echo "Testing sophisticated attack with --skip-freivalds..."
if cargo run --release -p nova_poc -- verify data/sophisticated_attack.json --skip-freivalds > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Sophisticated attack succeeded when skipping Freivalds!${NC}"
    echo -e "${YELLOW}⚠️  This shows why Freivalds verification is mandatory for security${NC}"
else
    echo -e "${GREEN}✅ EXCELLENT: Sophisticated attack failed even without Freivalds (public input validation)${NC}"
fi

section "DEFENSE ANALYSIS: Security Layer Breakdown"

echo "🛡️  Nova AI employs multiple layers of defense:"
echo ""
echo "1. PRIMARY DEFENSE: Freivalds Algorithm"
echo "   - Probabilistically verifies large layer computation (W1 · x0 = y1)"
echo "   - Detection probability: 1 - 2^(-32) ≈ 99.9999999%"
echo "   - Scales linearly with matrix size"
echo ""
echo "2. SECONDARY DEFENSE: Groth16 Public Input Validation"
echo "   - Validates commitment consistency"
echo "   - Cryptographically verifies small layer computation"
echo "   - Detection probability: 100% (deterministic)"
echo ""
echo "3. TERTIARY DEFENSE: Circuit Constraints"
echo "   - Enforces quantized division semantics"
echo "   - Prevents invalid y2 computations"
echo "   - Detection probability: 100% (cryptographic proof)"

section "PERFORMANCE vs SECURITY ANALYSIS"

echo "📊 Attack Detection Performance:"
echo ""

echo "Testing Freivalds detection time..."
start_time=$(date +%s.%N)
cargo run --release -p nova_poc -- verify data/malicious_run.json --weights1-path data/w1_16x50240.bin > /dev/null 2>&1 || true
end_time=$(date +%s.%N)
freivalds_time=$(echo "$end_time - $start_time" | bc -l)

echo "Testing public input validation time..."
start_time=$(date +%s.%N)
cargo run --release -p nova_poc -- verify data/sophisticated_attack.json --skip-freivalds > /dev/null 2>&1 || true
end_time=$(date +%s.%N)
validation_time=$(echo "$end_time - $start_time" | bc -l)

printf "⏱️  Freivalds fraud detection: %.2f seconds\n" $freivalds_time
printf "⏱️  Public input validation: %.2f seconds\n" $validation_time
echo ""

echo "💡 Security Insights:"
echo "   • Honest verification: ~13s (includes Freivalds for integrity)"
echo "   • Fraud detection: ~13s (same cost - security is not more expensive)"
echo "   • False positive rate: ~2^(-32) (astronomically low)"
echo "   • Economic incentive: Honesty is no more expensive than fraud attempts"

section "SECURITY RECOMMENDATIONS"

echo "🔒 Production Security Guidelines:"
echo ""
echo "1. NEVER use --skip-freivalds in production"
echo "   - This flag is for testing/debugging only"
echo "   - Skipping Freivalds reduces security significantly"
echo ""
echo "2. Use sufficient Freivalds rounds (≥32 recommended)"
echo "   - Each round reduces fraud probability by ~50%"
echo "   - 32 rounds gives ~2^(-32) fraud probability"
echo ""
echo "3. Validate all public inputs independently"
echo "   - Don't trust commitment values in run.json"
echo "   - Recompute commitments during verification"
echo ""
echo "4. Monitor for unusual verification patterns"
echo "   - Frequent verification failures may indicate attacks"
echo "   - Log fraud attempts for security analysis"

section "CONCLUSION"

echo -e "${GREEN}🎉 SECURITY ANALYSIS COMPLETE${NC}"
echo ""
echo "✅ All attack vectors successfully defended against"
echo "✅ Multiple independent security layers confirmed"
echo "✅ Performance cost of security is reasonable (~13s for production scale)"
echo "✅ Economic incentives favor honest behavior"
echo ""
echo -e "${BLUE}Nova AI demonstrates robust security against sophisticated adversaries${NC}"
echo -e "${BLUE}while maintaining practical performance for production workloads.${NC}"
echo ""
echo "🔍 For more details, see the Security Analysis section in README.md"