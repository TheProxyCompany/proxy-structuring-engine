# Best Effort Matching

Best Effort Matching is a strategy that enables forward progress in structured generation by accepting the largest valid prefix of a token that advances the state machine, even when the LLM generates extra or incorrect characters.

This allows the system to gracefully handle LLM overgeneration while maintaining schema compliance.
