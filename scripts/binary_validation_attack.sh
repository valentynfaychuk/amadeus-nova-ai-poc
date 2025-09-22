#!/bin/bash
# Binary Validation Attack: Demonstrating the discovered GKR vulnerability
# This script shows how padding corruption bypasses verification

echo "🔥 BINARY VALIDATION ATTACK - CRITICAL VULNERABILITY DEMONSTRATION"
echo "=================================================================="
echo ""
echo "This script demonstrates the discovered vulnerability where"
echo "binary corruptions in padding regions are not detected by the verifier."
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Generate a legitimate proof for testing
echo "📊 Generating legitimate baseline proof..."
./target/release/nova_poc demo --seed 12345 --m 16 --k 1024 > /dev/null 2>&1

if [ ! -f "proof_1024/gkr_proof.bin" ]; then
    echo "❌ Failed to generate baseline proof"
    exit 1
fi

echo "✅ Baseline proof generated"
echo ""

# Show the original proof structure
echo "🔍 Original proof structure:"
echo "   Size: $(stat -f%z proof_1024/gkr_proof.bin 2>/dev/null || stat -c%s proof_1024/gkr_proof.bin) bytes"
echo "   SHA256: $(sha256sum proof_1024/gkr_proof.bin | cut -d' ' -f1)"

# Test verification of original proof
echo ""
echo "🔧 Verifying original proof..."
if ./target/release/nova_poc verify --proof-path proof_1024/gkr_proof.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Original proof verifies correctly${NC}"
else
    echo -e "${RED}❌ Original proof verification failed${NC}"
    exit 1
fi

echo ""
echo "🚨 VULNERABILITY DEMONSTRATION:"
echo "================================"
echo ""

# Test 1: Padding region corruption
echo "🎯 Test 1: Padding region corruption"
cp proof_1024/gkr_proof.bin corrupted_padding.bin

# Corrupt a byte in the known padding region (around offset 500)
printf '\xFF' | dd of=corrupted_padding.bin bs=1 seek=500 count=1 conv=notrunc 2>/dev/null

echo "   Corrupted byte at offset 500 (padding region)"
echo "   New SHA256: $(sha256sum corrupted_padding.bin | cut -d' ' -f1)"

if ./target/release/nova_poc verify --proof-path corrupted_padding.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Corrupted proof in padding region VERIFIED!${NC}"
    echo -e "${RED}   This should have been rejected but was accepted${NC}"
else
    echo -e "${GREEN}✅ Corrupted proof correctly rejected${NC}"
fi

echo ""

# Test 2: Trailing data append
echo "🎯 Test 2: Trailing data append"
cp proof_1024/gkr_proof.bin trailing_data.bin

# Append malicious data to the end
echo "MALICIOUS_PAYLOAD_DATA_THAT_SHOULD_BE_DETECTED" >> trailing_data.bin

echo "   Appended 50 bytes of trailing data"
echo "   New size: $(stat -f%z trailing_data.bin 2>/dev/null || stat -c%s trailing_data.bin) bytes"
echo "   New SHA256: $(sha256sum trailing_data.bin | cut -d' ' -f1)"

if ./target/release/nova_poc verify --proof-path trailing_data.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Proof with trailing data VERIFIED!${NC}"
    echo -e "${RED}   Trailing data should be detected and rejected${NC}"
else
    echo -e "${GREEN}✅ Proof with trailing data correctly rejected${NC}"
fi

echo ""

# Test 3: Multiple padding corruptions
echo "🎯 Test 3: Multiple padding region corruptions"
cp proof_1024/gkr_proof.bin multi_corrupt.bin

# Corrupt multiple bytes in different padding regions
for offset in 100 200 500 1000 1500; do
    printf '\xDEADBEEF' | dd of=multi_corrupt.bin bs=1 seek=$offset count=4 conv=notrunc 2>/dev/null
done

echo "   Corrupted 4 bytes each at offsets: 100, 200, 500, 1000, 1500"
echo "   New SHA256: $(sha256sum multi_corrupt.bin | cut -d' ' -f1)"

if ./target/release/nova_poc verify --proof-path multi_corrupt.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Multiple corruptions VERIFIED!${NC}"
    echo -e "${RED}   Extensive corruption should definitely be detected${NC}"
else
    echo -e "${GREEN}✅ Multiple corruptions correctly rejected${NC}"
fi

echo ""

# Test 4: Strategic important data corruption
echo "🎯 Test 4: Strategic important data corruption (should fail)"
cp proof_1024/gkr_proof.bin important_corrupt.bin

# Corrupt what should be important data (near the beginning)
printf '\x00\x00\x00\x00' | dd of=important_corrupt.bin bs=1 seek=4 count=4 conv=notrunc 2>/dev/null

echo "   Corrupted 4 bytes at offset 4 (should be important data)"
echo "   New SHA256: $(sha256sum important_corrupt.bin | cut -d' ' -f1)"

if ./target/release/nova_poc verify --proof-path important_corrupt.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
    echo -e "${RED}❌ CRITICAL: Important data corruption VERIFIED!${NC}"
    echo -e "${RED}   This indicates even critical data isn't being validated${NC}"
else
    echo -e "${GREEN}✅ Important data corruption correctly rejected${NC}"
fi

echo ""
echo "📊 VULNERABILITY ANALYSIS:"
echo "=========================="
echo ""

# Analyze which regions are vulnerable
echo "🔍 Mapping vulnerable regions..."

vulnerable_count=0
total_tests=0

echo "Testing corruption at various offsets:"
for offset in $(seq 10 50 2000); do
    total_tests=$((total_tests + 1))

    cp proof_1024/gkr_proof.bin test_offset_${offset}.bin
    printf '\x42' | dd of=test_offset_${offset}.bin bs=1 seek=$offset count=1 conv=notrunc 2>/dev/null

    if ./target/release/nova_poc verify --proof-path test_offset_${offset}.bin --public-path proof_1024/public.json > /dev/null 2>&1; then
        vulnerable_count=$((vulnerable_count + 1))
        echo "   Offset $offset: VULNERABLE"
    fi

    rm -f test_offset_${offset}.bin
done

echo ""
echo "📈 Vulnerability Statistics:"
echo "   Total offsets tested: $total_tests"
echo "   Vulnerable offsets: $vulnerable_count"
echo "   Vulnerability rate: $(( vulnerable_count * 100 / total_tests ))%"

if [ $vulnerable_count -gt 0 ]; then
    echo ""
    echo -e "${RED}🚨 SECURITY ALERT: CRITICAL VULNERABILITY CONFIRMED${NC}"
    echo -e "${RED}   The GKR verifier accepts corrupted proofs in certain regions${NC}"
    echo -e "${RED}   This violates fundamental proof integrity assumptions${NC}"
    echo ""
    echo -e "${YELLOW}📋 REQUIRED ACTIONS:${NC}"
    echo "   1. Fix binary deserialization to validate all bytes"
    echo "   2. Add integrity checksums to proof format"
    echo "   3. Implement strict parsing mode"
    echo "   4. Add comprehensive binary validation tests"
    echo ""
    echo -e "${YELLOW}🔧 PROPOSED FIX:${NC}"
    echo "   Add this validation to deserialize_sumcheck_proof():"
    echo "   if cursor != data.len() {"
    echo "       return Err(GkrError::UnexpectedTrailingBytes);"
    echo "   }"
else
    echo ""
    echo -e "${GREEN}✅ NO VULNERABILITY DETECTED${NC}"
    echo "   All corruptions were properly rejected"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up test files..."
rm -f corrupted_padding.bin trailing_data.bin multi_corrupt.bin important_corrupt.bin

echo ""
echo "🔍 BINARY VALIDATION ATTACK COMPLETE"
echo ""
echo "This attack demonstrates the importance of complete binary validation"
echo "in cryptographic proof systems. Even 'harmless' padding corruptions"
echo "can undermine the fundamental integrity assumptions of the protocol."