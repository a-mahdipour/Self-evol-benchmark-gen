# Self-evol-benchmark-gen
*To continuously generate, answer, and semantically evaluate novel reasoning questions while adaptively adjusting difficulty, creating a self-evolving benchmark for AI performance.*

----

## Motivation
Static benchmarks saturate. Once models train against them, performance improvements no longer reflect genuine cognitive capability but memorization or distribution overfitting.
A self-evolving benchmark addresses this by:
*•	Continuously generating novel tasks
*•	Evaluating responses using a model-based or rule-based evaluator
*•	Tracking performance using an exponential moving average (EMA)
*•	Adapting task difficulty based on performance

The system becomes a closed-loop evaluation engine rather than a static test suite.

