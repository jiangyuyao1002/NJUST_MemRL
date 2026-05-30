/**
 * MemRL Memory Prompt Section
 *
 * Injects episodic hints and learned LTM rules into the system prompt,
 * enabling the agent to leverage cross-session RL-refined memory.
 */

/**
 * Build the MemRL memory section for the system prompt.
 *
 * @param episodicHints - Formatted episodic hints from EpisodicMemoryService.retrieve()
 * @param ltmRules      - Formatted LTM rule bullets from LongTermMemoryService.retrieve()
 * @returns Formatted prompt string, or empty string if both inputs are empty.
 */
export function getMemrlMemorySection(episodicHints: string, ltmRules: string): string {
	const parts: string[] = []

	if (episodicHints) {
		parts.push(episodicHints)
	}

	if (ltmRules) {
		parts.push(ltmRules)
	}

	if (parts.length === 0) return ""

	return (
		`\n\n## MemRL Adaptive Memory\n\n` +
		`The following memory was retrieved from past task experience. Use it to guide your approach:\n\n` +
		parts.join("\n\n")
	)
}
