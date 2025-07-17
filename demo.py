#!/usr/bin/env python3
"""
Demo script showing how to use the expert system core module.
This demonstrates the basic functionality of the rule-based expert system.
"""

from expert_system_core import ExpertSystem, Rule, Fact
from medical_knowledge_base import load_medical_knowledge_base

def main():
    print("=== Medical Diagnosis Expert System Demo ===\n")
    
    # Initialize expert system
    print("1. Initializing Expert System...")
    es = ExpertSystem()
    
    # Load medical knowledge base
    print("2. Loading Medical Knowledge Base...")
    rules = load_medical_knowledge_base()
    for rule in rules:
        es.add_rule(rule)
    
    print(f"   - Loaded {len(es.rules)} rules")
    print(f"   - System initialized with {len(es.facts)} facts\n")
    
    # Demo scenario: Patient with flu symptoms
    print("3. Adding Patient Symptoms...")
    patient_symptoms = [
        Fact("fever", 1.0, "patient report"),
        Fact("headache", 0.9, "patient report"),
        Fact("body_ache", 0.8, "patient report"),
        Fact("sore_throat", 0.7, "patient report"),
        Fact("fatigue", 0.9, "patient report")
    ]
    
    for symptom in patient_symptoms:
        es.add_fact(symptom)
        print(f"   - Added: {symptom}")
    
    print(f"\n   Total facts in working memory: {len(es.facts)}")
    
    # Forward chaining inference
    print("\n4. Running Forward Chaining Inference...")
    trace = es.forward_chain()
    
    print(f"   - Inference completed in {len(trace)} steps")
    print(f"   - Total facts after inference: {len(es.facts)}")
    
    # Show inference trace
    print("\n5. Inference Trace:")
    for i, step in enumerate(trace, 1):
        print(f"   Step {i}: Applied rule {step['rule_applied']}")
        print(f"           {step['rule_description']}")
        print(f"           New facts: {', '.join(step['new_facts'])}")
        print()
    
    # Show diagnoses
    diagnoses = [fact for fact in es.facts if fact in ['flu', 'covid', 'cold', 'allergy']]
    recommendations = [fact for fact in es.facts if fact.startswith('recommend_')]
    
    print("6. Diagnostic Results:")
    if diagnoses:
        print("   Diagnoses:")
        for diagnosis in diagnoses:
            confidence = es.facts[diagnosis].confidence
            print(f"   - {diagnosis.upper()} (confidence: {confidence:.2f})")
    
    if recommendations:
        print("   Recommendations:")
        for rec in recommendations:
            print(f"   - {rec.replace('_', ' ').title()}")
    
    print()
    
    # Backward chaining demo
    print("7. Testing Backward Chaining...")
    goal = "flu"
    proven, bt_trace = es.backward_chain(goal)
    
    print(f"   - Goal '{goal}' was {'PROVEN' if proven else 'NOT PROVEN'}")
    print(f"   - Reasoning steps: {len(bt_trace)}")
    
    # Explanation demo
    print("\n8. Explanation Facility:")
    if es.fact_exists(goal):
        explanation = es.explain_fact(goal)
        print(f"   How was '{goal}' derived?")
        for item in explanation:
            if "explanation" in item:
                print(f"   - {item['explanation']}")
            elif "derived_by" in item:
                print(f"   - Derived by rule: {item['derived_by']}")
                print(f"     Description: {item['rule_description']}")
                print(f"     Confidence: {item['confidence']:.2f}")
    
    # System state summary
    print(f"\n9. Final System State:")
    print(f"   - Rules in knowledge base: {len(es.get_rules())}")
    print(f"   - Facts in working memory: {len(es.get_facts())}")
    print(f"   - Inference trace entries: {len(es.inference_trace)}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()