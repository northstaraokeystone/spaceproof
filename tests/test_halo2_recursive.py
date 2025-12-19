"""Tests for Halo2 infinite recursive ZK proofs."""


class TestHalo2Config:
    """Tests for Halo2 configuration."""

    def test_load_halo2_config(self) -> None:
        """Test loading Halo2 configuration."""
        from src.halo2_recursive import load_halo2_config

        config = load_halo2_config()
        assert config is not None
        assert "proof_system" in config
        assert config["proof_system"] == "halo2"
        assert "circuit_size" in config
        assert "no_trusted_setup" in config
        assert config["no_trusted_setup"] is True

    def test_halo2_constants(self) -> None:
        """Test Halo2 constants are correctly defined."""
        from src.halo2_recursive import (
            HALO2_CIRCUIT_SIZE,
            HALO2_PROOF_TIME_MS,
            HALO2_VERIFY_TIME_MS,
            HALO2_RESILIENCE_TARGET,
            HALO2_NO_TRUSTED_SETUP,
            HALO2_IPA_COMMITMENT,
        )

        assert HALO2_CIRCUIT_SIZE == 2**24
        assert HALO2_PROOF_TIME_MS == 150
        assert HALO2_VERIFY_TIME_MS == 3
        assert HALO2_RESILIENCE_TARGET == 0.97
        assert HALO2_NO_TRUSTED_SETUP is True
        assert HALO2_IPA_COMMITMENT is True


class TestHalo2Circuit:
    """Tests for Halo2 circuit generation."""

    def test_generate_circuit(self) -> None:
        """Test Halo2 circuit generation."""
        from src.halo2_recursive import generate_halo2_circuit

        circuit = generate_halo2_circuit()

        assert circuit is not None
        assert "circuit_id" in circuit
        assert "constraints" in circuit
        assert "gates" in circuit

    def test_circuit_size_param(self) -> None:
        """Test circuit size parameter."""
        from src.halo2_recursive import generate_halo2_circuit, HALO2_CIRCUIT_SIZE

        circuit = generate_halo2_circuit(circuit_size=HALO2_CIRCUIT_SIZE)

        assert circuit["constraints"] == HALO2_CIRCUIT_SIZE


class TestHalo2Proof:
    """Tests for Halo2 proof generation and verification."""

    def test_generate_proof(self) -> None:
        """Test Halo2 proof generation."""
        from src.halo2_recursive import generate_halo2_circuit, generate_halo2_proof

        circuit = generate_halo2_circuit()
        proof = generate_halo2_proof(
            circuit_id=circuit["circuit_id"],
            public_inputs=[1, 2, 3],
            private_inputs=[10, 20, 30],
        )

        assert proof is not None
        assert "proof_id" in proof
        assert "commitment" in proof
        assert "proof_time_ms" in proof

    def test_verify_proof(self) -> None:
        """Test Halo2 proof verification."""
        from src.halo2_recursive import (
            generate_halo2_circuit,
            generate_halo2_proof,
            verify_halo2_proof,
        )

        circuit = generate_halo2_circuit()
        proof = generate_halo2_proof(
            circuit_id=circuit["circuit_id"],
            public_inputs=[1, 2, 3],
            private_inputs=[10, 20, 30],
        )
        verification = verify_halo2_proof(
            proof_id=proof["proof_id"],
            circuit_id=circuit["circuit_id"],
            public_inputs=[1, 2, 3],
        )

        assert verification is not None
        assert "valid" in verification
        assert verification["valid"] is True
        assert "verify_time_ms" in verification


class TestHalo2Recursion:
    """Tests for Halo2 recursive proof capabilities."""

    def test_accumulate_proofs(self) -> None:
        """Test proof accumulation using IPA."""
        from src.halo2_recursive import accumulate_proofs

        proof_ids = ["proof_1", "proof_2", "proof_3"]
        result = accumulate_proofs(proof_ids)

        assert result is not None
        assert "accumulated_proof" in result
        assert "accumulator" in result
        assert "proofs_accumulated" in result
        assert result["proofs_accumulated"] == 3
        assert "accumulation_valid" in result

    def test_recursive_verify(self) -> None:
        """Test recursive verification."""
        from src.halo2_recursive import accumulate_proofs, recursive_verify

        proof_ids = ["proof_1", "proof_2", "proof_3"]
        accumulation = accumulate_proofs(proof_ids)

        result = recursive_verify(
            accumulated_proof=accumulation["accumulated_proof"],
            depth=3,
        )

        assert result is not None
        assert "valid" in result
        assert "accumulated_depth" in result
        assert result["accumulated_depth"] == 3

    def test_generate_recursive_proof(self) -> None:
        """Test recursive proof generation."""
        from src.halo2_recursive import generate_recursive_proof

        result = generate_recursive_proof(
            depth=3,
            base_inputs=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )

        assert result is not None
        assert "proof_chain" in result
        assert "accumulator" in result
        assert "compression_ratio" in result
        assert len(result["proof_chain"]) == 3

    def test_verify_recursive_proof(self) -> None:
        """Test recursive proof verification."""
        from src.halo2_recursive import (
            generate_recursive_proof,
            verify_recursive_proof,
        )

        recursive = generate_recursive_proof(
            depth=3,
            base_inputs=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        )

        result = verify_recursive_proof(
            proof_chain=recursive["proof_chain"],
            accumulator=recursive["accumulator"],
        )

        assert result is not None
        assert "valid" in result
        assert result["valid"] is True
        assert "accumulated_depth" in result


class TestHalo2Attestation:
    """Tests for Halo2 attestation."""

    def test_create_attestation(self) -> None:
        """Test Halo2 attestation creation."""
        from src.halo2_recursive import create_halo2_attestation

        result = create_halo2_attestation(
            enclave_id="test_enclave",
            code_hash="test_code_hash",
            config_hash="test_config_hash",
            recursion_depth=0,
        )

        assert result is not None
        assert "attestation_id" in result
        assert "proof_id" in result
        assert "claims" in result
        assert "public_inputs" in result
        assert "metadata" in result

    def test_verify_attestation(self) -> None:
        """Test Halo2 attestation verification."""
        from src.halo2_recursive import (
            create_halo2_attestation,
            verify_halo2_attestation,
        )

        attestation = create_halo2_attestation(
            enclave_id="test_enclave",
            code_hash="test_code_hash",
            config_hash="test_config_hash",
            recursion_depth=0,
        )

        result = verify_halo2_attestation(
            attestation_id=attestation["attestation_id"],
            enclave_id="test_enclave",
        )

        assert result is not None
        assert "valid" in result
        assert result["valid"] is True


class TestHalo2Benchmark:
    """Tests for Halo2 benchmarking."""

    def test_benchmark(self) -> None:
        """Test Halo2 benchmark."""
        from src.halo2_recursive import benchmark_halo2

        result = benchmark_halo2(iterations=5)

        assert result is not None
        assert "proof_time_ms" in result
        assert "verify_time_ms" in result
        assert "throughput_proofs_per_sec" in result
        assert "throughput_verifies_per_sec" in result

        # Check statistics structure
        assert "avg" in result["proof_time_ms"]
        assert "min" in result["proof_time_ms"]
        assert "max" in result["proof_time_ms"]
        assert "std" in result["proof_time_ms"]


class TestHalo2Comparison:
    """Tests for Halo2 comparison with other systems."""

    def test_compare_to_plonk(self) -> None:
        """Test comparison with PLONK."""
        from src.halo2_recursive import compare_to_plonk

        result = compare_to_plonk()

        assert result is not None
        assert "halo2" in result
        assert "plonk" in result
        assert "groth16" in result
        assert "halo2_advantages" in result
        assert "recommendation" in result

    def test_halo2_features(self) -> None:
        """Test Halo2 feature comparison."""
        from src.halo2_recursive import compare_to_plonk

        result = compare_to_plonk()

        halo2 = result["halo2"]
        assert halo2["no_trusted_setup"] is True
        assert halo2["infinite_recursion"] is True
        assert halo2["ipa_commitment"] is True


class TestHalo2Audit:
    """Tests for Halo2 audit functionality."""

    def test_run_audit(self) -> None:
        """Test Halo2 audit run."""
        from src.halo2_recursive import run_halo2_audit

        result = run_halo2_audit(attestation_count=3)

        assert result is not None
        assert "attestations_created" in result
        assert "verifications_passed" in result
        assert "verification_rate" in result
        assert "resilience" in result
        assert "resilience_target" in result
        assert "overall_validated" in result

    def test_audit_resilience_target(self) -> None:
        """Test that audit checks resilience target."""
        from src.halo2_recursive import run_halo2_audit, HALO2_RESILIENCE_TARGET

        result = run_halo2_audit(attestation_count=3)

        assert result["resilience_target"] == HALO2_RESILIENCE_TARGET
        assert "resilience_target_met" in result
