# Member 3 - Red-Team Agent

## Purpose
Documents Member 3's attack generation, Gemma inference, feedback memory, and pending Member 4 integration.

## Files
- attack_generator.py: Dummy and Gemma attack-case generation.
- gemma_client.py: Local Ollama inference client.
- feedback_memory.py: Persistent feedback storage and retrieval.

## Dummy Mode
Use generate_attack_cases() for fixed test cases without model inference.

## Gemma Mode
Use generate_attack_cases(limit=2, use_gemma=True) for Gemma-powered generation.

## Adaptive Generation
Use generate_adaptive_variants() with an existing saved feedback ID.

## Feedback Storage
Default location: artifacts/attack_feedback.jsonl

## Member 4 Integration
The evaluation harness and its agreed interface were not present when this document was created. Confirm the feedback format and run-ID contract before integration.

## Tests
Run python -m unittest tests.test_red_team -v
Run python -m unittest discover -s tests -v
