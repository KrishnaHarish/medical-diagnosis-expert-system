## Overview

This PR implements a complete Rule-Based Expert System for Medical Diagnosis using Forward and Backward Chaining algorithms as a mini project suitable for 10 marks. The system demonstrates classical AI reasoning techniques from Russell & Norvig's "Artificial Intelligence: A Modern Approach" without using any machine learning.

## Features Implemented

### Core AI Algorithms
- **Forward Chaining (Data-Driven)**: Starts with symptoms and derives all possible diagnoses
- **Backward Chaining (Goal-Driven)**: Tests if a specific diagnosis can be proven from given symptoms
- **Working Memory Management**: Tracks facts and reasoning chains during inference
- **Explanation Facility**: Provides step-by-step reasoning transparency

### Knowledge Base
- 15 medical IF-THEN rules covering common conditions
- Confidence levels and explanations for each rule
- Covers diagnoses: Common Cold, Influenza, Strep Throat, Migraine, Allergic Reactions, etc.
- Treatment recommendations linked to diagnoses

### User Interface
- Interactive CLI with menu-driven system
- Symptom input with multiple formats (numbers or names)
- Real-time diagnosis with both inference methods
- Comprehensive explanation viewing

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │ Inference Engine│    │ Knowledge Base  │
│     (CLI)       │◄──►│   - Forward     │◄──►│   - Rules       │
│                 │    │   - Backward    │    │   - Facts       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
         │              ┌─────────────────┐              
         │              │ Working Memory  │              
         │              │ - Facts         │              
         │              │ - Reasoning     │              
         └──────────────┤   Chain         │              
                        └─────────────────┘              
                                 │                       
                        ┌─────────────────┐              
                        │  Explanation    │              
                        │    Module       │              
                        └─────────────────┘              
```

## Files Added

- `main.py` - Main entry point and system launcher
- `knowledge_base.py` - Medical rules and facts database (15 rules, 28 symptoms)
- `inference_engine.py` - Forward/backward chaining algorithms implementation
- `user_interface.py` - Interactive CLI with comprehensive menu system
- `test_system.py` - Comprehensive test suite validating all algorithms
- `README.md` - Complete project documentation with usage examples
- `project_report.md` - Technical report (2 pages) explaining architecture and algorithms
- `.gitignore` - Python-specific ignore patterns

## Usage Examples

### Running the System
```bash
# Interactive mode (default)
python main.py

# Demonstration mode
python main.py --demo

# Run tests
python test_system.py
```

### Forward Chaining Example
```
Input: runny_nose, sneezing, sore_throat
Output: common_cold, rest, drink_fluids
Reasoning: Rule R001 → Rule R011 (confidence: 0.85 → 0.90)
```

### Backward Chaining Example
```
Goal: migraine
Input: severe_headache, nausea, light_sensitivity
Result: PROVEN (confidence: 0.70)
Proof Chain: severe_headache → nausea → light_sensitivity → migraine
```

## Technical Highlights

### Forward Chaining Algorithm
- Implements data-driven inference with iterative rule application
- Monotonic reasoning with fact accumulation
- Conflict resolution through rule ordering
- Guaranteed termination with finite rule sets

### Backward Chaining Algorithm
- Goal-driven inference with recursive proof search
- Depth-first search with cycle detection
- Confidence propagation through proof chains
- Efficient pruning of impossible goals

### Explanation System
- Step-by-step reasoning traces
- Rule application tracking
- Confidence level propagation
- Human-readable explanations for all inferences

## Testing

The system includes comprehensive tests covering:
- Knowledge base integrity (15 rules, 28 symptoms, 19 diagnoses)
- Forward chaining with various symptom combinations
- Backward chaining with successful and failed proofs
- Explanation facility functionality
- All tests pass successfully

## Educational Value

This project demonstrates:
- **Classical AI Reasoning**: Rule-based expert systems
- **Search Algorithms**: Forward and backward chaining
- **Knowledge Representation**: IF-THEN rules with confidence
- **Inference Mechanisms**: Logical reasoning without ML
- **Explanation Systems**: Transparent AI decision-making

## Disclaimer

This system is for educational purposes only and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## Project Stats
- **Total Lines**: 1,477
- **Rules**: 15 medical diagnostic rules
- **Symptoms**: 28 different medical symptoms
- **Diagnoses**: 19 possible outcomes
- **Test Coverage**: 100% of core algorithms tested

<!-- START COPILOT CODING AGENT TIPS -->
---

💡 You can make Copilot smarter by setting up custom instructions, customizing its development environment and configuring Model Context Protocol (MCP) servers. Learn more [Copilot coding agent tips](https://gh.io/copilot-coding-agent-tips) in the docs.
