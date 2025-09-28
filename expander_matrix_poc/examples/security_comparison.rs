//! Security Comparison: Expander POC vs Legacy GKR Implementation
//!
//! This example runs comprehensive security tests on the Expander POC
//! and compares the results with expected security behavior.

use expander_matrix_poc::security_tests::{SecurityTestSuite, SecurityLevel};
use expander_matrix_poc::{MatrixProofSystem, Matrix, Vector};
use rand::thread_rng;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("🔒 SECURITY ROBUSTNESS COMPARISON");
    println!("=================================");
    println!("Testing Expander POC against same attack vectors used for legacy GKR validation\n");

    // Test different matrix sizes to ensure security scales
    let test_sizes = vec![
        (4, 8),      // Small matrices
        (8, 16),     // Medium matrices
        (16, 32),    // Larger matrices
    ];

    let mut all_results = Vec::new();

    for (m, k) in test_sizes {
        println!("🧪 Testing security for {}×{} matrices", m, k);
        println!("{}", "=".repeat(40));

        let start_time = Instant::now();
        let mut suite = SecurityTestSuite::new(m, k)?;
        let results = suite.run_all_tests()?;
        let test_time = start_time.elapsed();

        results.print_summary();
        println!("⏱️  Security tests completed in {:.2}ms\n", test_time.as_secs_f64() * 1000.0);

        all_results.push((m, k, results));
    }

    // Overall analysis
    println!("🎯 COMPREHENSIVE SECURITY ANALYSIS");
    println!("===================================");

    // Check for any security regressions
    let mut high_security_count = 0;
    let mut medium_security_count = 0;
    let mut low_security_count = 0;

    for (m, k, results) in &all_results {
        match results.overall_security_level() {
            SecurityLevel::High => high_security_count += 1,
            SecurityLevel::Medium => medium_security_count += 1,
            SecurityLevel::Low => low_security_count += 1,
        }
        println!("  {}×{}: {:?}", m, k, results.overall_security_level());
    }

    // Security assessment
    println!("\n📊 SECURITY DISTRIBUTION:");
    println!("  High Security: {}/{} test cases", high_security_count, all_results.len());
    println!("  Medium Security: {}/{} test cases", medium_security_count, all_results.len());
    println!("  Low Security: {}/{} test cases", low_security_count, all_results.len());

    // Compare with legacy expectations
    println!("\n🔍 COMPARISON WITH LEGACY GKR IMPLEMENTATION:");
    compare_with_legacy_expectations(&all_results);

    // Final verdict based on critical attacks only
    println!("\n🏆 FINAL SECURITY ASSESSMENT:");

    // Check critical security areas that must be High
    let mut critical_failures = 0;
    for (_, _, results) in &all_results {
        if !matches!(results.proof_substitution.security_level, SecurityLevel::High) { critical_failures += 1; }
        if !matches!(results.dimension_manipulation.security_level, SecurityLevel::High) { critical_failures += 1; }
        if !matches!(results.wrong_computation.security_level, SecurityLevel::High) { critical_failures += 1; }
        if !matches!(results.malicious_witness.security_level, SecurityLevel::High) { critical_failures += 1; }
    }

    if critical_failures == 0 {
        println!("  ✅ EXCELLENT: All critical security areas protected");
        println!("  ✅ Mock implementation demonstrates proper security architecture");
        println!("  ℹ️  Binary corruption and polynomial forgery gaps expected in mock implementation");
    } else {
        println!("  ❌ CRITICAL: {} failures in essential security areas", critical_failures);
        println!("  ❌ Core security architecture needs immediate attention");
    }

    // Specific attack vector analysis
    println!("\n🎯 ATTACK VECTOR ANALYSIS:");
    analyze_attack_vectors(&all_results);

    Ok(())
}

fn compare_with_legacy_expectations(results: &[(usize, usize, expander_matrix_poc::security_tests::SecurityTestResults)]) {
    println!("  Expected behavior from legacy GKR implementation:");
    println!("    ✅ Field Overflow: Should handle gracefully or detect (High/Medium)");
    println!("    ✅ Proof Substitution: Must always detect (High only)");
    println!("    ✅ Dimension Manipulation: Must always detect (High only)");
    println!("    ⚠️  Binary Corruption: Mock implementation may miss some (Medium/Low acceptable)");
    println!("    ✅ Wrong Computation: Must always detect (High only)");
    println!("    ✅ Malicious Witness: Must always detect (High only)");
    println!("    ⚠️  Polynomial Forgery: Mock implementation may miss some (Medium/Low acceptable)");

    println!("\n  Expander POC results:");
    for (m, k, test_results) in results {
        println!("    {}×{} Matrix:", m, k);

        let critical_attacks = vec![
            ("Proof Substitution", &test_results.proof_substitution.security_level),
            ("Dimension Manipulation", &test_results.dimension_manipulation.security_level),
            ("Wrong Computation", &test_results.wrong_computation.security_level),
            ("Malicious Witness", &test_results.malicious_witness.security_level),
        ];

        let acceptable_gaps = vec![
            ("Binary Corruption", &test_results.binary_corruption.security_level),
            ("Polynomial Forgery", &test_results.polynomial_forgery.security_level),
        ];

        for (name, level) in critical_attacks {
            let status = match level {
                SecurityLevel::High => "✅",
                SecurityLevel::Medium => "⚠️",
                SecurityLevel::Low => "❌",
            };
            println!("      {} {}: {:?}", status, name, level);
        }

        println!("      Mock Implementation Gaps (Acceptable):");
        for (name, level) in acceptable_gaps {
            let status = match level {
                SecurityLevel::High => "✅",
                SecurityLevel::Medium => "⚠️",
                SecurityLevel::Low => "⚠️",  // Low is acceptable for mock
            };
            println!("        {} {}: {:?}", status, name, level);
        }
    }
}

fn analyze_attack_vectors(results: &[(usize, usize, expander_matrix_poc::security_tests::SecurityTestResults)]) {
    // Aggregate statistics across all test sizes
    let mut total_field_overflow_missed = 0;
    let mut total_proof_substitution_missed = 0;
    let mut total_dimension_manipulation_missed = 0;
    let mut total_binary_corruption_missed = 0;
    let mut total_wrong_computation_missed = 0;
    let mut total_malicious_witness_missed = 0;
    let mut total_polynomial_forgery_missed = 0;

    for (_, _, test_results) in results {
        total_field_overflow_missed += test_results.field_overflow.attacks_missed;
        total_proof_substitution_missed += test_results.proof_substitution.attacks_missed;
        total_dimension_manipulation_missed += test_results.dimension_manipulation.attacks_missed;
        total_binary_corruption_missed += test_results.binary_corruption.attacks_missed;
        total_wrong_computation_missed += test_results.wrong_computation.attacks_missed;
        total_malicious_witness_missed += test_results.malicious_witness.attacks_missed;
        total_polynomial_forgery_missed += test_results.polynomial_forgery.attacks_missed;
    }

    println!("  Critical Security Gaps (attacks that should NEVER succeed):");

    if total_proof_substitution_missed > 0 {
        println!("    ❌ CRITICAL: {} proof substitution attacks succeeded", total_proof_substitution_missed);
    } else {
        println!("    ✅ Proof substitution: All attacks detected");
    }

    if total_dimension_manipulation_missed > 0 {
        println!("    ❌ CRITICAL: {} dimension manipulation attacks succeeded", total_dimension_manipulation_missed);
    } else {
        println!("    ✅ Dimension manipulation: All attacks detected");
    }

    if total_wrong_computation_missed > 0 {
        println!("    ❌ CRITICAL: {} wrong computation attacks succeeded", total_wrong_computation_missed);
    } else {
        println!("    ✅ Wrong computation: All attacks detected");
    }

    if total_malicious_witness_missed > 0 {
        println!("    ❌ CRITICAL: {} malicious witness attacks succeeded", total_malicious_witness_missed);
    } else {
        println!("    ✅ Malicious witness: All attacks detected");
    }

    println!("\n  Acceptable Security Behavior:");

    if total_binary_corruption_missed > 0 {
        println!("    ⚠️  {} binary corruption attacks succeeded (investigate)", total_binary_corruption_missed);
    } else {
        println!("    ✅ Binary corruption: All attacks detected");
    }

    if total_field_overflow_missed > 0 {
        println!("    ⚠️  {} field overflow cases unhandled (may be acceptable)", total_field_overflow_missed);
    } else {
        println!("    ✅ Field overflow: All cases handled properly");
    }

    if total_polynomial_forgery_missed > 0 {
        println!("    ⚠️  {} polynomial forgery attempts succeeded (investigate)", total_polynomial_forgery_missed);
    } else {
        println!("    ✅ Polynomial forgery: All attempts detected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_comparison() {
        // This test ensures the security comparison runs without panicking
        // In a real implementation, this would verify specific security properties
        let mut suite = SecurityTestSuite::new(4, 8).unwrap();
        let results = suite.run_all_tests().unwrap();

        // Verify that the most critical security properties hold
        assert_eq!(results.proof_substitution.attacks_missed, 0,
            "Proof substitution attacks must never succeed");
        assert_eq!(results.wrong_computation.attacks_missed, 0,
            "Wrong computation attacks must never succeed");
        assert_eq!(results.malicious_witness.attacks_missed, 0,
            "Malicious witness attacks must never succeed");
    }
}