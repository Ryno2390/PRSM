#!/usr/bin/env python3
"""
Chemistry Test Data and Evaluation Metrics
Comprehensive test datasets and metrics for chemistry reasoning benchmarks

This module provides:
1. Curated chemistry test datasets with ground truth
2. Evaluation metrics for chemistry reasoning quality
3. Test data generation utilities
4. Domain-specific validation functions

Usage:
    from prsm.evaluation.chemistry_test_data import ChemistryTestDataset
    
    dataset = ChemistryTestDataset()
    test_cases = dataset.get_reaction_prediction_tests()
    metrics = dataset.evaluate_prediction(prediction, ground_truth)
"""

import json
import csv
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import random
import numpy as np
import re

from pydantic import BaseModel, Field
from prsm.nwtn.chemistry_hybrid_executor import ReactionConditions


class ChemistryTestCategory(str, Enum):
    """Categories of chemistry test cases"""
    REACTION_PREDICTION = "reaction_prediction"
    THERMODYNAMICS = "thermodynamics"
    KINETICS = "kinetics"
    MOLECULAR_PROPERTIES = "molecular_properties"
    MECHANISM_ANALYSIS = "mechanism_analysis"
    EQUILIBRIUM = "equilibrium"
    CATALYSIS = "catalysis"
    ORGANIC_SYNTHESIS = "organic_synthesis"
    INORGANIC_CHEMISTRY = "inorganic_chemistry"
    ANALYTICAL_CHEMISTRY = "analytical_chemistry"


class DifficultyLevel(str, Enum):
    """Difficulty levels for test cases"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EvaluationMetric(str, Enum):
    """Evaluation metrics for chemistry reasoning"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    CHEMICAL_ACCURACY = "chemical_accuracy"
    MECHANISM_CORRECTNESS = "mechanism_correctness"
    THERMODYNAMIC_CONSISTENCY = "thermodynamic_consistency"
    REASONING_QUALITY = "reasoning_quality"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


@dataclass
class ChemistryTestCase:
    """Individual chemistry test case with ground truth"""
    test_id: str
    category: ChemistryTestCategory
    difficulty: DifficultyLevel
    question: str
    ground_truth: Dict[str, Any]
    context: Dict[str, Any]
    evaluation_criteria: List[str]
    expected_concepts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ChemistryEvaluationResult:
    """Results of chemistry reasoning evaluation"""
    test_id: str
    predicted_answer: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metrics: Dict[EvaluationMetric, float]
    detailed_analysis: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enum keys to strings for JSON serialization
        result["metrics"] = {k.value if hasattr(k, 'value') else str(k): v for k, v in result["metrics"].items()}
        return result


class ChemistryTestDataset:
    """
    Comprehensive chemistry test dataset with evaluation capabilities
    
    This class provides curated test cases for various chemistry domains
    and evaluation metrics to assess reasoning quality.
    """
    
    def __init__(self):
        self.test_cases: Dict[str, ChemistryTestCase] = {}
        self.evaluation_functions: Dict[EvaluationMetric, callable] = {}
        self._initialize_test_cases()
        self._initialize_evaluation_functions()
    
    def _initialize_test_cases(self):
        """Initialize comprehensive test case library"""
        
        # Reaction Prediction Tests
        self.test_cases.update(self._create_reaction_prediction_tests())
        
        # Thermodynamics Tests
        self.test_cases.update(self._create_thermodynamics_tests())
        
        # Kinetics Tests
        self.test_cases.update(self._create_kinetics_tests())
        
        # Molecular Properties Tests
        self.test_cases.update(self._create_molecular_properties_tests())
        
        # Mechanism Analysis Tests
        self.test_cases.update(self._create_mechanism_analysis_tests())
        
        # Equilibrium Tests
        self.test_cases.update(self._create_equilibrium_tests())
        
        # Catalysis Tests
        self.test_cases.update(self._create_catalysis_tests())
        
        # Organic Synthesis Tests
        self.test_cases.update(self._create_organic_synthesis_tests())
        
        # Inorganic Chemistry Tests
        self.test_cases.update(self._create_inorganic_chemistry_tests())
        
        # Analytical Chemistry Tests
        self.test_cases.update(self._create_analytical_chemistry_tests())
    
    def _create_reaction_prediction_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create reaction prediction test cases"""
        tests = {}
        
        # Basic reaction prediction
        tests["rxn_001"] = ChemistryTestCase(
            test_id="rxn_001",
            category=ChemistryTestCategory.REACTION_PREDICTION,
            difficulty=DifficultyLevel.BASIC,
            question="Predict the products of the reaction: 2Na + Cl2 →",
            ground_truth={
                "will_react": True,
                "products": ["2NaCl"],
                "reaction_type": "synthesis",
                "balanced_equation": "2Na + Cl2 → 2NaCl",
                "gibbs_free_energy": -411.2,
                "spontaneous": True
            },
            context={
                "reactants": ["Na", "Cl2"],
                "conditions": {"temperature": 298.15, "pressure": 1.0},
                "phase": "solid + gas → solid"
            },
            evaluation_criteria=[
                "correct_products",
                "balanced_equation",
                "thermodynamic_assessment",
                "reaction_type_identification"
            ],
            expected_concepts=[
                "ionic_bonding",
                "electron_transfer",
                "oxidation_reduction",
                "lattice_energy"
            ]
        )
        
        # Intermediate reaction prediction
        tests["rxn_002"] = ChemistryTestCase(
            test_id="rxn_002",
            category=ChemistryTestCategory.REACTION_PREDICTION,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Predict the products and determine feasibility: CH4 + 2O2 → (combustion)",
            ground_truth={
                "will_react": True,
                "products": ["CO2", "2H2O"],
                "reaction_type": "combustion",
                "balanced_equation": "CH4 + 2O2 → CO2 + 2H2O",
                "gibbs_free_energy": -818.0,
                "enthalpy_change": -890.3,
                "spontaneous": True,
                "requires_activation": True,
                "activation_energy": 430.0
            },
            context={
                "reactants": ["CH4", "O2"],
                "conditions": {"temperature": 298.15, "pressure": 1.0},
                "reaction_environment": "gas_phase"
            },
            evaluation_criteria=[
                "correct_products",
                "balanced_equation",
                "thermodynamic_analysis",
                "kinetic_considerations",
                "activation_energy_assessment"
            ],
            expected_concepts=[
                "combustion_reaction",
                "oxidation_states",
                "enthalpy_of_combustion",
                "activation_energy",
                "reaction_coordinate"
            ]
        )
        
        # Advanced reaction prediction
        tests["rxn_003"] = ChemistryTestCase(
            test_id="rxn_003",
            category=ChemistryTestCategory.REACTION_PREDICTION,
            difficulty=DifficultyLevel.ADVANCED,
            question="Predict the major product of the SN2 reaction: CH3CH2Br + OH⁻ → (in polar aprotic solvent)",
            ground_truth={
                "will_react": True,
                "major_product": "CH3CH2OH",
                "minor_products": ["CH2=CH2"],
                "reaction_type": "SN2",
                "mechanism": "concerted_substitution",
                "stereochemistry": "inversion",
                "rate_law": "rate = k[CH3CH2Br][OH⁻]",
                "favored_by": ["polar_aprotic_solvent", "good_nucleophile", "primary_carbon"]
            },
            context={
                "substrate": "CH3CH2Br",
                "nucleophile": "OH⁻",
                "solvent": "polar_aprotic",
                "temperature": 298.15,
                "leaving_group": "Br⁻"
            },
            evaluation_criteria=[
                "correct_major_product",
                "mechanism_identification",
                "stereochemistry_prediction",
                "rate_law_determination",
                "solvent_effect_analysis"
            ],
            expected_concepts=[
                "SN2_mechanism",
                "nucleophilic_substitution",
                "backside_attack",
                "leaving_group_ability",
                "solvent_effects"
            ]
        )
        
        return tests
    
    def _create_thermodynamics_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create thermodynamics test cases"""
        tests = {}
        
        tests["thermo_001"] = ChemistryTestCase(
            test_id="thermo_001",
            category=ChemistryTestCategory.THERMODYNAMICS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Calculate the Gibbs free energy change for: H2(g) + 1/2O2(g) → H2O(l) at 298K",
            ground_truth={
                "delta_g": -237.1,
                "delta_h": -285.8,
                "delta_s": -163.2,
                "spontaneous": True,
                "calculation_method": "delta_g = delta_h - T*delta_s",
                "units": "kJ/mol"
            },
            context={
                "temperature": 298.15,
                "pressure": 1.0,
                "standard_conditions": True,
                "phase_changes": ["gas", "gas", "liquid"]
            },
            evaluation_criteria=[
                "correct_delta_g_value",
                "proper_calculation_method",
                "spontaneity_assessment",
                "unit_consistency"
            ],
            expected_concepts=[
                "gibbs_free_energy",
                "enthalpy",
                "entropy",
                "thermodynamic_spontaneity"
            ]
        )
        
        return tests
    
    def _create_kinetics_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create kinetics test cases"""
        tests = {}
        
        tests["kinetics_001"] = ChemistryTestCase(
            test_id="kinetics_001",
            category=ChemistryTestCategory.KINETICS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Determine the rate law for the reaction: 2NO(g) + O2(g) → 2NO2(g) given experimental data",
            ground_truth={
                "rate_law": "rate = k[NO]²[O2]",
                "overall_order": 3,
                "order_NO": 2,
                "order_O2": 1,
                "rate_constant_units": "L²/(mol²·s)",
                "mechanism_type": "elementary_reaction"
            },
            context={
                "experimental_data": [
                    {"[NO]": 0.10, "[O2]": 0.10, "rate": 2.5e-5},
                    {"[NO]": 0.20, "[O2]": 0.10, "rate": 1.0e-4},
                    {"[NO]": 0.10, "[O2]": 0.20, "rate": 5.0e-5}
                ],
                "temperature": 298.15,
                "method": "initial_rates"
            },
            evaluation_criteria=[
                "correct_rate_law",
                "correct_reaction_orders",
                "proper_data_analysis",
                "rate_constant_units"
            ],
            expected_concepts=[
                "rate_law",
                "reaction_order",
                "rate_constant",
                "method_of_initial_rates"
            ]
        )
        
        return tests
    
    def _create_molecular_properties_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create molecular properties test cases"""
        tests = {}
        
        tests["molprop_001"] = ChemistryTestCase(
            test_id="molprop_001",
            category=ChemistryTestCategory.MOLECULAR_PROPERTIES,
            difficulty=DifficultyLevel.BASIC,
            question="Determine the molecular geometry and polarity of NH3",
            ground_truth={
                "molecular_geometry": "trigonal_pyramidal",
                "electron_geometry": "tetrahedral",
                "bond_angles": "approximately 107°",
                "hybridization": "sp³",
                "polar": True,
                "dipole_moment": 1.46,
                "lone_pairs": 1
            },
            context={
                "molecule": "NH3",
                "central_atom": "N",
                "valence_electrons": 8,
                "bonding_pairs": 3,
                "lone_pairs": 1
            },
            evaluation_criteria=[
                "correct_geometry",
                "correct_hybridization",
                "polarity_assessment",
                "bond_angle_prediction"
            ],
            expected_concepts=[
                "VSEPR_theory",
                "hybridization",
                "molecular_polarity",
                "lone_pair_effects"
            ]
        )
        
        return tests
    
    def _create_mechanism_analysis_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create mechanism analysis test cases"""
        tests = {}
        
        tests["mechanism_001"] = ChemistryTestCase(
            test_id="mechanism_001",
            category=ChemistryTestCategory.MECHANISM_ANALYSIS,
            difficulty=DifficultyLevel.ADVANCED,
            question="Analyze the mechanism for the acid-catalyzed hydration of alkenes",
            ground_truth={
                "mechanism_type": "electrophilic_addition",
                "steps": [
                    "protonation_of_alkene",
                    "carbocation_formation",
                    "nucleophilic_attack_by_water",
                    "deprotonation"
                ],
                "intermediate": "carbocation",
                "regioselectivity": "Markovnikov",
                "rate_determining_step": "carbocation_formation",
                "stereochemistry": "no_stereospecificity"
            },
            context={
                "substrate": "alkene",
                "catalyst": "H⁺",
                "nucleophile": "H2O",
                "product": "alcohol",
                "conditions": "aqueous_acid"
            },
            evaluation_criteria=[
                "correct_mechanism_steps",
                "intermediate_identification",
                "regioselectivity_prediction",
                "rate_determining_step"
            ],
            expected_concepts=[
                "electrophilic_addition",
                "carbocation_stability",
                "Markovnikov_rule",
                "reaction_intermediates"
            ]
        )
        
        return tests
    
    def _create_equilibrium_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create equilibrium test cases"""
        tests = {}
        
        tests["equilibrium_001"] = ChemistryTestCase(
            test_id="equilibrium_001",
            category=ChemistryTestCategory.EQUILIBRIUM,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Calculate the equilibrium constant for: N2(g) + 3H2(g) ⇌ 2NH3(g) at 400°C",
            ground_truth={
                "equilibrium_constant": 0.5,
                "temperature": 673.15,
                "delta_g_standard": 32.9,
                "calculation_method": "K = exp(-ΔG°/RT)",
                "units": "dimensionless",
                "equilibrium_lies": "left"
            },
            context={
                "reaction": "N2(g) + 3H2(g) ⇌ 2NH3(g)",
                "temperature": 673.15,
                "pressure": 1.0,
                "delta_g_formation_NH3": -16.45
            },
            evaluation_criteria=[
                "correct_equilibrium_constant",
                "proper_calculation_method",
                "equilibrium_position_assessment",
                "temperature_effect_understanding"
            ],
            expected_concepts=[
                "equilibrium_constant",
                "gibbs_free_energy",
                "Le_Chatelier_principle",
                "temperature_dependence"
            ]
        )
        
        return tests
    
    def _create_catalysis_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create catalysis test cases"""
        tests = {}
        
        tests["catalysis_001"] = ChemistryTestCase(
            test_id="catalysis_001",
            category=ChemistryTestCategory.CATALYSIS,
            difficulty=DifficultyLevel.ADVANCED,
            question="Explain how a catalyst increases the rate of the reaction: H2O2 → H2O + 1/2O2",
            ground_truth={
                "mechanism": "alternate_pathway",
                "activation_energy_reduction": "significant",
                "catalyst_examples": ["catalase", "MnO2", "KI"],
                "rate_enhancement": "10^6 to 10^17",
                "thermodynamics_unchanged": True,
                "catalyst_recovery": True
            },
            context={
                "reaction": "H2O2 decomposition",
                "uncatalyzed_activation_energy": 75.0,
                "catalyzed_activation_energy": 25.0,
                "catalyst_type": "heterogeneous"
            },
            evaluation_criteria=[
                "mechanism_explanation",
                "activation_energy_concept",
                "catalyst_properties",
                "rate_enhancement_understanding"
            ],
            expected_concepts=[
                "catalysis",
                "activation_energy",
                "reaction_coordinate",
                "enzyme_kinetics"
            ]
        )
        
        return tests
    
    def _create_organic_synthesis_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create organic synthesis test cases"""
        tests = {}
        
        tests["organic_001"] = ChemistryTestCase(
            test_id="organic_001",
            category=ChemistryTestCategory.ORGANIC_SYNTHESIS,
            difficulty=DifficultyLevel.EXPERT,
            question="Design a synthesis route for aspirin from salicylic acid",
            ground_truth={
                "starting_material": "salicylic_acid",
                "target_product": "aspirin",
                "key_reaction": "acetylation",
                "reagents": ["acetic_anhydride", "phosphoric_acid"],
                "mechanism": "nucleophilic_acyl_substitution",
                "yield": "high",
                "purification": "recrystallization"
            },
            context={
                "functional_groups": ["phenol", "carboxylic_acid"],
                "protection_needed": False,
                "reaction_conditions": "mild",
                "industrial_process": True
            },
            evaluation_criteria=[
                "correct_synthesis_route",
                "appropriate_reagents",
                "mechanism_understanding",
                "practical_considerations"
            ],
            expected_concepts=[
                "acetylation",
                "nucleophilic_substitution",
                "pharmaceutical_synthesis",
                "reaction_optimization"
            ]
        )
        
        return tests
    
    def _create_inorganic_chemistry_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create inorganic chemistry test cases"""
        tests = {}
        
        tests["inorganic_001"] = ChemistryTestCase(
            test_id="inorganic_001",
            category=ChemistryTestCategory.INORGANIC_CHEMISTRY,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Predict the crystal structure and magnetic properties of [Fe(CN)6]³⁻",
            ground_truth={
                "geometry": "octahedral",
                "crystal_field": "strong_field",
                "electron_configuration": "t2g⁵",
                "magnetic_property": "paramagnetic",
                "unpaired_electrons": 1,
                "color": "yellow-green",
                "ligand_field_strength": "high"
            },
            context={
                "central_metal": "Fe³⁺",
                "ligands": "CN⁻",
                "coordination_number": 6,
                "d_electron_count": 5
            },
            evaluation_criteria=[
                "correct_geometry",
                "crystal_field_analysis",
                "magnetic_property_prediction",
                "electronic_configuration"
            ],
            expected_concepts=[
                "coordination_chemistry",
                "crystal_field_theory",
                "magnetic_properties",
                "ligand_field_strength"
            ]
        )
        
        return tests
    
    def _create_analytical_chemistry_tests(self) -> Dict[str, ChemistryTestCase]:
        """Create analytical chemistry test cases"""
        tests = {}
        
        tests["analytical_001"] = ChemistryTestCase(
            test_id="analytical_001",
            category=ChemistryTestCategory.ANALYTICAL_CHEMISTRY,
            difficulty=DifficultyLevel.INTERMEDIATE,
            question="Calculate the pH of a 0.1 M solution of acetic acid (Ka = 1.8 × 10⁻⁵)",
            ground_truth={
                "ph": 2.87,
                "calculation_method": "weak_acid_approximation",
                "percent_dissociation": 1.3,
                "h_plus_concentration": 1.34e-3,
                "assumptions_valid": True
            },
            context={
                "concentration": 0.1,
                "acid_constant": 1.8e-5,
                "temperature": 298.15,
                "ionic_strength": "low"
            },
            evaluation_criteria=[
                "correct_ph_calculation",
                "appropriate_method",
                "assumption_validation",
                "significant_figures"
            ],
            expected_concepts=[
                "weak_acid_equilibrium",
                "pH_calculation",
                "acid_dissociation_constant",
                "chemical_equilibrium"
            ]
        )
        
        return tests
    
    def _initialize_evaluation_functions(self):
        """Initialize evaluation functions for different metrics"""
        
        self.evaluation_functions = {
            EvaluationMetric.ACCURACY: self._evaluate_accuracy,
            EvaluationMetric.PRECISION: self._evaluate_precision,
            EvaluationMetric.RECALL: self._evaluate_recall,
            EvaluationMetric.F1_SCORE: self._evaluate_f1_score,
            EvaluationMetric.CHEMICAL_ACCURACY: self._evaluate_chemical_accuracy,
            EvaluationMetric.MECHANISM_CORRECTNESS: self._evaluate_mechanism_correctness,
            EvaluationMetric.THERMODYNAMIC_CONSISTENCY: self._evaluate_thermodynamic_consistency,
            EvaluationMetric.REASONING_QUALITY: self._evaluate_reasoning_quality,
            EvaluationMetric.CONFIDENCE_CALIBRATION: self._evaluate_confidence_calibration
        }
    
    def get_test_cases(self, 
                      category: Optional[ChemistryTestCategory] = None,
                      difficulty: Optional[DifficultyLevel] = None,
                      limit: Optional[int] = None) -> List[ChemistryTestCase]:
        """Get test cases filtered by category and difficulty"""
        
        filtered_cases = []
        
        for test_case in self.test_cases.values():
            if category and test_case.category != category:
                continue
            if difficulty and test_case.difficulty != difficulty:
                continue
            filtered_cases.append(test_case)
        
        if limit:
            filtered_cases = filtered_cases[:limit]
        
        return filtered_cases
    
    def get_test_case(self, test_id: str) -> Optional[ChemistryTestCase]:
        """Get specific test case by ID"""
        return self.test_cases.get(test_id)
    
    def evaluate_prediction(self, 
                          test_case: ChemistryTestCase,
                          prediction: Dict[str, Any],
                          reasoning_trace: List[Dict[str, Any]] = None,
                          confidence: float = 0.0) -> ChemistryEvaluationResult:
        """Evaluate a prediction against ground truth"""
        
        metrics = {}
        detailed_analysis = {}
        
        # Calculate all applicable metrics
        for metric, eval_func in self.evaluation_functions.items():
            try:
                score = eval_func(prediction, test_case.ground_truth, reasoning_trace)
                metrics[metric] = score
            except Exception as e:
                metrics[metric] = 0.0
                detailed_analysis[f"{metric.value}_error"] = str(e)
        
        # Detailed analysis based on test category
        if test_case.category == ChemistryTestCategory.REACTION_PREDICTION:
            detailed_analysis.update(self._analyze_reaction_prediction(prediction, test_case))
        elif test_case.category == ChemistryTestCategory.THERMODYNAMICS:
            detailed_analysis.update(self._analyze_thermodynamics(prediction, test_case))
        elif test_case.category == ChemistryTestCategory.KINETICS:
            detailed_analysis.update(self._analyze_kinetics(prediction, test_case))
        
        return ChemistryEvaluationResult(
            test_id=test_case.test_id,
            predicted_answer=prediction,
            ground_truth=test_case.ground_truth,
            metrics=metrics,
            detailed_analysis=detailed_analysis,
            reasoning_trace=reasoning_trace or [],
            confidence_score=confidence
        )
    
    def _evaluate_accuracy(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                          reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Calculate overall accuracy"""
        
        correct_predictions = 0
        total_predictions = 0
        
        for key, true_value in ground_truth.items():
            if key in prediction:
                predicted_value = prediction[key]
                
                if isinstance(true_value, bool):
                    correct_predictions += int(predicted_value == true_value)
                elif isinstance(true_value, (int, float)):
                    # Allow 5% tolerance for numerical values
                    tolerance = abs(true_value * 0.05)
                    correct_predictions += int(abs(predicted_value - true_value) <= tolerance)
                elif isinstance(true_value, str):
                    correct_predictions += int(predicted_value.lower() == true_value.lower())
                elif isinstance(true_value, list):
                    # Check if lists have same elements (order independent)
                    correct_predictions += int(set(predicted_value) == set(true_value))
                
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _evaluate_precision(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                           reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Calculate precision for classification tasks"""
        
        # Extract boolean predictions
        true_positives = 0
        false_positives = 0
        
        for key, pred_value in prediction.items():
            if isinstance(pred_value, bool) and pred_value:
                ground_truth_value = ground_truth.get(key, False)
                if ground_truth_value:
                    true_positives += 1
                else:
                    false_positives += 1
        
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    def _evaluate_recall(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                        reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Calculate recall for classification tasks"""
        
        # Extract boolean predictions
        true_positives = 0
        false_negatives = 0
        
        for key, true_value in ground_truth.items():
            if isinstance(true_value, bool) and true_value:
                pred_value = prediction.get(key, False)
                if pred_value:
                    true_positives += 1
                else:
                    false_negatives += 1
        
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    def _evaluate_f1_score(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                          reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Calculate F1 score"""
        
        precision = self._evaluate_precision(prediction, ground_truth, reasoning_trace)
        recall = self._evaluate_recall(prediction, ground_truth, reasoning_trace)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _evaluate_chemical_accuracy(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                                   reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Evaluate accuracy of chemical predictions"""
        
        chemical_accuracy = 0.0
        
        # Check chemical formula correctness
        if "products" in prediction and "products" in ground_truth:
            pred_products = set(prediction["products"])
            true_products = set(ground_truth["products"])
            
            if pred_products == true_products:
                chemical_accuracy += 0.4
            elif pred_products & true_products:  # Partial overlap
                chemical_accuracy += 0.2
        
        # Check reaction type
        if "reaction_type" in prediction and "reaction_type" in ground_truth:
            if prediction["reaction_type"] == ground_truth["reaction_type"]:
                chemical_accuracy += 0.2
        
        # Check thermodynamic consistency
        if "spontaneous" in prediction and "spontaneous" in ground_truth:
            if prediction["spontaneous"] == ground_truth["spontaneous"]:
                chemical_accuracy += 0.2
        
        # Check mechanism correctness
        if "mechanism" in prediction and "mechanism" in ground_truth:
            if prediction["mechanism"] == ground_truth["mechanism"]:
                chemical_accuracy += 0.2
        
        return chemical_accuracy
    
    def _evaluate_mechanism_correctness(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                                      reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Evaluate correctness of reaction mechanism"""
        
        mechanism_score = 0.0
        
        # Check mechanism steps
        if "steps" in prediction and "steps" in ground_truth:
            pred_steps = prediction["steps"]
            true_steps = ground_truth["steps"]
            
            # Check if key steps are present
            correct_steps = 0
            for step in true_steps:
                if step in pred_steps:
                    correct_steps += 1
            
            mechanism_score += (correct_steps / len(true_steps)) * 0.5
        
        # Check intermediates
        if "intermediate" in prediction and "intermediate" in ground_truth:
            if prediction["intermediate"] == ground_truth["intermediate"]:
                mechanism_score += 0.3
        
        # Check rate determining step
        if "rate_determining_step" in prediction and "rate_determining_step" in ground_truth:
            if prediction["rate_determining_step"] == ground_truth["rate_determining_step"]:
                mechanism_score += 0.2
        
        return mechanism_score
    
    def _evaluate_thermodynamic_consistency(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                                          reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Evaluate thermodynamic consistency"""
        
        consistency_score = 0.0
        
        # Check Gibbs free energy prediction
        if "delta_g" in prediction and "delta_g" in ground_truth:
            pred_g = prediction["delta_g"]
            true_g = ground_truth["delta_g"]
            
            relative_error = abs(pred_g - true_g) / abs(true_g) if true_g != 0 else float('inf')
            if relative_error < 0.1:  # Within 10%
                consistency_score += 0.4
            elif relative_error < 0.2:  # Within 20%
                consistency_score += 0.2
        
        # Check spontaneity prediction
        if "spontaneous" in prediction and "spontaneous" in ground_truth:
            if prediction["spontaneous"] == ground_truth["spontaneous"]:
                consistency_score += 0.3
        
        # Check temperature dependence
        if "temperature_effect" in prediction and "temperature_effect" in ground_truth:
            if prediction["temperature_effect"] == ground_truth["temperature_effect"]:
                consistency_score += 0.3
        
        return consistency_score
    
    def _evaluate_reasoning_quality(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                                   reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Evaluate quality of reasoning trace"""
        
        if not reasoning_trace:
            return 0.0
        
        quality_score = 0.0
        
        # Check reasoning length (should be substantive)
        if len(reasoning_trace) >= 3:
            quality_score += 0.2
        
        # Check for scientific terminology
        scientific_terms = [
            "thermodynamic", "kinetic", "activation energy", "gibbs free energy",
            "catalyst", "equilibrium", "enthalpy", "entropy", "mechanism",
            "bond", "electron", "orbital", "molecular", "reaction"
        ]
        
        trace_text = " ".join(
            step.get("description", "").lower() for step in reasoning_trace
        )
        
        term_count = sum(1 for term in scientific_terms if term in trace_text)
        quality_score += min(term_count / 10, 0.4)  # Up to 0.4 for terminology
        
        # Check for logical structure
        if any("therefore" in step.get("description", "").lower() for step in reasoning_trace):
            quality_score += 0.2
        
        # Check for quantitative analysis
        if any(char.isdigit() for char in trace_text):
            quality_score += 0.2
        
        return quality_score
    
    def _evaluate_confidence_calibration(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any], 
                                       reasoning_trace: List[Dict[str, Any]] = None) -> float:
        """Evaluate confidence calibration"""
        
        confidence = prediction.get("confidence", 0.5)
        accuracy = self._evaluate_accuracy(prediction, ground_truth, reasoning_trace)
        
        # Good calibration means confidence matches accuracy
        calibration_error = abs(confidence - accuracy)
        
        # Return score that's higher when calibration error is lower
        return max(0.0, 1.0 - calibration_error)
    
    def _analyze_reaction_prediction(self, prediction: Dict[str, Any], test_case: ChemistryTestCase) -> Dict[str, Any]:
        """Detailed analysis for reaction prediction"""
        
        analysis = {}
        
        # Product analysis
        if "products" in prediction and "products" in test_case.ground_truth:
            pred_products = set(prediction["products"])
            true_products = set(test_case.ground_truth["products"])
            
            analysis["products_correct"] = pred_products == true_products
            analysis["products_overlap"] = len(pred_products & true_products)
            analysis["missing_products"] = list(true_products - pred_products)
            analysis["extra_products"] = list(pred_products - true_products)
        
        # Thermodynamic analysis
        if "gibbs_free_energy" in prediction and "gibbs_free_energy" in test_case.ground_truth:
            pred_g = prediction["gibbs_free_energy"]
            true_g = test_case.ground_truth["gibbs_free_energy"]
            
            analysis["gibbs_error"] = abs(pred_g - true_g)
            analysis["gibbs_relative_error"] = abs(pred_g - true_g) / abs(true_g) if true_g != 0 else float('inf')
        
        return analysis
    
    def _analyze_thermodynamics(self, prediction: Dict[str, Any], test_case: ChemistryTestCase) -> Dict[str, Any]:
        """Detailed analysis for thermodynamics"""
        
        analysis = {}
        
        # Energy analysis
        for energy_type in ["delta_g", "delta_h", "delta_s"]:
            if energy_type in prediction and energy_type in test_case.ground_truth:
                pred_val = prediction[energy_type]
                true_val = test_case.ground_truth[energy_type]
                
                analysis[f"{energy_type}_error"] = abs(pred_val - true_val)
                analysis[f"{energy_type}_relative_error"] = abs(pred_val - true_val) / abs(true_val) if true_val != 0 else float('inf')
        
        return analysis
    
    def _analyze_kinetics(self, prediction: Dict[str, Any], test_case: ChemistryTestCase) -> Dict[str, Any]:
        """Detailed analysis for kinetics"""
        
        analysis = {}
        
        # Rate law analysis
        if "rate_law" in prediction and "rate_law" in test_case.ground_truth:
            pred_rate_law = prediction["rate_law"].replace(" ", "").lower()
            true_rate_law = test_case.ground_truth["rate_law"].replace(" ", "").lower()
            
            analysis["rate_law_correct"] = pred_rate_law == true_rate_law
        
        # Reaction order analysis
        if "overall_order" in prediction and "overall_order" in test_case.ground_truth:
            analysis["order_correct"] = prediction["overall_order"] == test_case.ground_truth["overall_order"]
        
        return analysis
    
    def export_test_cases(self, filename: str, format: str = "json"):
        """Export test cases to file"""
        
        data = {
            "test_cases": [case.to_dict() for case in self.test_cases.values()],
            "metadata": {
                "total_cases": len(self.test_cases),
                "categories": list(set(case.category for case in self.test_cases.values())),
                "difficulty_levels": list(set(case.difficulty for case in self.test_cases.values()))
            }
        }
        
        filepath = Path(filename)
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == "csv":
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'test_id', 'category', 'difficulty', 'question', 'ground_truth', 'context'
                ])
                writer.writeheader()
                for case in self.test_cases.values():
                    writer.writerow({
                        'test_id': case.test_id,
                        'category': case.category.value,
                        'difficulty': case.difficulty.value,
                        'question': case.question,
                        'ground_truth': json.dumps(case.ground_truth),
                        'context': json.dumps(case.context)
                    })
    
    def import_test_cases(self, filename: str):
        """Import test cases from file"""
        
        filepath = Path(filename)
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for case_data in data.get("test_cases", []):
                case = ChemistryTestCase(
                    test_id=case_data["test_id"],
                    category=ChemistryTestCategory(case_data["category"]),
                    difficulty=DifficultyLevel(case_data["difficulty"]),
                    question=case_data["question"],
                    ground_truth=case_data["ground_truth"],
                    context=case_data["context"],
                    evaluation_criteria=case_data["evaluation_criteria"],
                    expected_concepts=case_data["expected_concepts"]
                )
                self.test_cases[case.test_id] = case
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        
        stats = {
            "total_test_cases": len(self.test_cases),
            "categories": {},
            "difficulty_levels": {},
            "evaluation_metrics": len(self.evaluation_functions)
        }
        
        for case in self.test_cases.values():
            # Category distribution
            category = case.category.value
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Difficulty distribution
            difficulty = case.difficulty.value
            stats["difficulty_levels"][difficulty] = stats["difficulty_levels"].get(difficulty, 0) + 1
        
        return stats
    
    def create_benchmark_suite(self, 
                             categories: List[ChemistryTestCategory] = None,
                             difficulties: List[DifficultyLevel] = None,
                             size: int = 50) -> List[ChemistryTestCase]:
        """Create a balanced benchmark suite"""
        
        if categories is None:
            categories = list(ChemistryTestCategory)
        if difficulties is None:
            difficulties = list(DifficultyLevel)
        
        suite = []
        cases_per_category = size // len(categories)
        
        for category in categories:
            category_cases = self.get_test_cases(category=category)
            
            # Balance by difficulty
            for difficulty in difficulties:
                difficulty_cases = [c for c in category_cases if c.difficulty == difficulty]
                if difficulty_cases:
                    selected = random.sample(
                        difficulty_cases, 
                        min(cases_per_category // len(difficulties), len(difficulty_cases))
                    )
                    suite.extend(selected)
        
        return suite[:size]  # Ensure we don't exceed requested size


# Factory functions
def create_chemistry_test_dataset() -> ChemistryTestDataset:
    """Create a chemistry test dataset instance"""
    return ChemistryTestDataset()


def load_chemistry_test_data(filename: str) -> ChemistryTestDataset:
    """Load chemistry test data from file"""
    dataset = ChemistryTestDataset()
    dataset.import_test_cases(filename)
    return dataset


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset = create_chemistry_test_dataset()
    
    # Get statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:")
    print(f"Total test cases: {stats['total_test_cases']}")
    print(f"Categories: {stats['categories']}")
    print(f"Difficulty levels: {stats['difficulty_levels']}")
    
    # Create benchmark suite
    benchmark_suite = dataset.create_benchmark_suite(size=20)
    print(f"\nBenchmark suite created with {len(benchmark_suite)} test cases")
    
    # Test evaluation
    test_case = dataset.get_test_case("rxn_001")
    if test_case:
        # Mock prediction
        prediction = {
            "will_react": True,
            "products": ["2NaCl"],
            "reaction_type": "synthesis",
            "spontaneous": True
        }
        
        result = dataset.evaluate_prediction(test_case, prediction, confidence=0.9)
        print(f"\nEvaluation result for {test_case.test_id}:")
        print(f"Accuracy: {result.metrics[EvaluationMetric.ACCURACY]:.3f}")
        print(f"Chemical accuracy: {result.metrics[EvaluationMetric.CHEMICAL_ACCURACY]:.3f}")
    
    # Export test cases
    dataset.export_test_cases("chemistry_test_cases.json")
    print("\nTest cases exported to chemistry_test_cases.json")