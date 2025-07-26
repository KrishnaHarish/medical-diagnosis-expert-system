"""
Comprehensive test suite for the expert system core functionality.
Tests all classes and methods to validate correct implementation.
"""
import unittest
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expert_system_core import ExpertSystem, Rule, Fact
from medical_knowledge_base import load_medical_knowledge_base


class TestRule(unittest.TestCase):
    """Test cases for the Rule class."""
    
    def test_rule_creation(self):
        """Test creating a valid rule."""
        rule = Rule("R1", ["fever", "headache"], ["possible_flu"], "Test rule")
        self.assertEqual(rule.name, "R1")
        self.assertEqual(rule.antecedents, ["fever", "headache"])
        self.assertEqual(rule.consequents, ["possible_flu"])
        self.assertEqual(rule.description, "Test rule")
    
    def test_rule_str_representation(self):
        """Test string representation of a rule."""
        rule = Rule("R1", ["fever", "headache"], ["possible_flu"], "Test rule")
        expected = "Rule R1: IF fever AND headache THEN possible_flu"
        self.assertEqual(str(rule), expected)
    
    def test_rule_validation(self):
        """Test rule validation during creation."""
        # Test empty name
        with self.assertRaises(ValueError):
            Rule("", ["fever"], ["flu"])
        
        # Test empty antecedents
        with self.assertRaises(ValueError):
            Rule("R1", [], ["flu"])
        
        # Test empty consequents
        with self.assertRaises(ValueError):
            Rule("R1", ["fever"], [])
        
        # Test invalid antecedents type
        with self.assertRaises(TypeError):
            Rule("R1", "fever", ["flu"])
        
        # Test invalid consequents type
        with self.assertRaises(TypeError):
            Rule("R1", ["fever"], "flu")


class TestFact(unittest.TestCase):
    """Test cases for the Fact class."""
    
    def test_fact_creation(self):
        """Test creating a valid fact."""
        fact = Fact("fever", 0.9, "user input")
        self.assertEqual(fact.statement, "fever")
        self.assertEqual(fact.confidence, 0.9)
        self.assertEqual(fact.source, "user input")
    
    def test_fact_defaults(self):
        """Test fact creation with default values."""
        fact = Fact("fever")
        self.assertEqual(fact.statement, "fever")
        self.assertEqual(fact.confidence, 1.0)
        self.assertEqual(fact.source, "")
    
    def test_fact_str_representation(self):
        """Test string representation of a fact."""
        fact = Fact("fever", 0.9, "user input")
        expected = "fever [conf=0.90, src=user input]"
        self.assertEqual(str(fact), expected)
    
    def test_fact_validation(self):
        """Test fact validation during creation."""
        # Test empty statement
        with self.assertRaises(ValueError):
            Fact("", 1.0, "test")
        
        # Test invalid confidence range
        with self.assertRaises(ValueError):
            Fact("fever", 1.5, "test")
        
        with self.assertRaises(ValueError):
            Fact("fever", -0.1, "test")
        
        # Test invalid types
        with self.assertRaises(TypeError):
            Fact(123, 1.0, "test")
        
        with self.assertRaises(TypeError):
            Fact("fever", "high", "test")
        
        with self.assertRaises(TypeError):
            Fact("fever", 1.0, 123)


class TestExpertSystem(unittest.TestCase):
    """Test cases for the ExpertSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.es = ExpertSystem()
        self.rule1 = Rule("R1", ["fever", "headache"], ["possible_flu"], "Basic flu symptoms")
        self.rule2 = Rule("R2", ["possible_flu", "fatigue"], ["flu"], "Confirmed flu")
        self.fact1 = Fact("fever", 1.0, "user input")
        self.fact2 = Fact("headache", 0.8, "user input")
        self.fact3 = Fact("fatigue", 0.9, "user input")
    
    def test_initialization(self):
        """Test expert system initialization."""
        es = ExpertSystem()
        self.assertEqual(len(es.rules), 0)
        self.assertEqual(len(es.facts), 0)
        self.assertEqual(len(es.inference_trace), 0)
    
    def test_add_rule(self):
        """Test adding rules to the knowledge base."""
        self.es.add_rule(self.rule1)
        self.assertEqual(len(self.es.rules), 1)
        self.assertTrue(self.es.rule_exists("R1"))
        self.assertEqual(self.es.rules["R1"], self.rule1)
    
    def test_add_fact(self):
        """Test adding facts to working memory."""
        self.es.add_fact(self.fact1)
        self.assertEqual(len(self.es.facts), 1)
        self.assertTrue(self.es.fact_exists("fever"))
        self.assertEqual(self.es.facts["fever"], self.fact1)
    
    def test_clear_methods(self):
        """Test clearing rules and facts."""
        self.es.add_rule(self.rule1)
        self.es.add_fact(self.fact1)
        
        # Test clear_facts
        self.es.clear_facts()
        self.assertEqual(len(self.es.facts), 0)
        self.assertEqual(len(self.es.rules), 1)  # Rules should remain
        
        # Test clear_rules
        self.es.clear_rules()
        self.assertEqual(len(self.es.rules), 0)
    
    def test_get_methods(self):
        """Test getter methods."""
        self.es.add_rule(self.rule1)
        self.es.add_fact(self.fact1)
        
        rules = self.es.get_rules()
        facts = self.es.get_facts()
        
        self.assertEqual(len(rules), 1)
        self.assertEqual(len(facts), 1)
        self.assertIn("R1", rules)
        self.assertIn("fever", facts)
        
        # Test that returned copies don't affect original
        rules.clear()
        facts.clear()
        self.assertEqual(len(self.es.rules), 1)
        self.assertEqual(len(self.es.facts), 1)
    
    def test_forward_chaining(self):
        """Test forward chaining inference."""
        # Set up rules and facts
        self.es.add_rule(self.rule1)
        self.es.add_rule(self.rule2)
        self.es.add_fact(self.fact1)  # fever
        self.es.add_fact(self.fact2)  # headache
        
        # Run forward chaining
        trace = self.es.forward_chain()
        
        # Should have inferred possible_flu
        self.assertTrue(self.es.fact_exists("possible_flu"))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["rule_applied"], "R1")
        
        # Add fatigue and run again
        self.es.add_fact(self.fact3)
        trace = self.es.forward_chain()
        
        # Should have inferred flu
        self.assertTrue(self.es.fact_exists("flu"))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["rule_applied"], "R2")
    
    def test_backward_chaining(self):
        """Test backward chaining inference."""
        # Set up rules and facts
        self.es.add_rule(self.rule1)
        self.es.add_rule(self.rule2)
        self.es.add_fact(self.fact1)  # fever
        self.es.add_fact(self.fact2)  # headache
        self.es.add_fact(self.fact3)  # fatigue
        
        # Try to prove flu
        proven, trace = self.es.backward_chain("flu")
        
        self.assertTrue(proven)
        self.assertTrue(self.es.fact_exists("flu"))
        self.assertGreater(len(trace), 0)
        
        # Try to prove something that can't be proven
        proven, trace = self.es.backward_chain("covid")
        self.assertFalse(proven)
    
    def test_explain_fact(self):
        """Test fact explanation functionality."""
        # Set up rules and facts
        self.es.add_rule(self.rule1)
        self.es.add_fact(self.fact1)  # fever
        self.es.add_fact(self.fact2)  # headache
        
        # Run forward chaining to create derived facts
        self.es.forward_chain()
        
        # Explain a derived fact
        explanation = self.es.explain_fact("possible_flu")
        self.assertGreater(len(explanation), 0)
        self.assertIn("fact", explanation[0])
        self.assertIn("derived_by", explanation[0])
        
        # Explain a non-existent fact
        explanation = self.es.explain_fact("nonexistent")
        self.assertEqual(len(explanation), 1)
        self.assertIn("explanation", explanation[0])
        
        # Explain a directly provided fact
        explanation = self.es.explain_fact("fever")
        self.assertEqual(len(explanation), 1)
        self.assertIn("explanation", explanation[0])
    
    def test_validation_in_expert_system(self):
        """Test validation in ExpertSystem methods."""
        # Test add_rule validation
        with self.assertRaises(TypeError):
            self.es.add_rule("not a rule")
        
        # Test add_fact validation
        with self.assertRaises(TypeError):
            self.es.add_fact("not a fact")


class TestMedicalKnowledgeBase(unittest.TestCase):
    """Test cases for the medical knowledge base."""
    
    def test_load_medical_knowledge_base(self):
        """Test loading the medical knowledge base."""
        rules = load_medical_knowledge_base()
        self.assertGreater(len(rules), 0)
        
        # Check that all loaded items are Rule objects
        for rule in rules:
            self.assertIsInstance(rule, Rule)
            self.assertIsInstance(rule.name, str)
            self.assertIsInstance(rule.antecedents, list)
            self.assertIsInstance(rule.consequents, list)
            self.assertIsInstance(rule.description, str)
    
    def test_knowledge_base_integration(self):
        """Test integration with the expert system."""
        es = ExpertSystem()
        rules = load_medical_knowledge_base()
        
        # Add all rules
        for rule in rules:
            es.add_rule(rule)
        
        # Test that rules were added correctly
        self.assertEqual(len(es.rules), len(rules))
        
        # Test with some common symptoms
        es.add_fact(Fact("fever", 1.0, "user input"))
        es.add_fact(Fact("headache", 1.0, "user input"))
        es.add_fact(Fact("body_ache", 1.0, "user input"))
        
        # Run inference
        trace = es.forward_chain()
        
        # Should have made some inferences
        self.assertGreater(len(es.facts), 3)  # More than the initial 3 facts


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_complete_diagnosis_workflow(self):
        """Test a complete diagnosis workflow."""
        es = ExpertSystem()
        
        # Load knowledge base
        rules = load_medical_knowledge_base()
        for rule in rules:
            es.add_rule(rule)
        
        # Add flu symptoms
        es.add_fact(Fact("fever", 1.0, "user input"))
        es.add_fact(Fact("headache", 1.0, "user input"))
        es.add_fact(Fact("body_ache", 1.0, "user input"))
        es.add_fact(Fact("sore_throat", 1.0, "user input"))
        es.add_fact(Fact("fatigue", 1.0, "user input"))
        
        # Run forward chaining
        forward_trace = es.forward_chain()
        
        # Should diagnose flu
        self.assertTrue(es.fact_exists("flu"))
        
        # Test backward chaining for flu
        proven, backward_trace = es.backward_chain("flu")
        self.assertTrue(proven)
        
        # Test explanation
        explanation = es.explain_fact("flu")
        self.assertGreater(len(explanation), 0)
        
        # Test that we can clear and start over
        es.clear_facts()
        self.assertEqual(len(es.facts), 0)
        self.assertFalse(es.fact_exists("flu"))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)