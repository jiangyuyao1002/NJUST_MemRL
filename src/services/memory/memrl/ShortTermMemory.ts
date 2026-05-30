/**
 * ShortTermMemory
 *
 * Single-task in-memory step history.
 * Trims oldest entries when the total character count exceeds STM_MAX_CHARS.
 */

import { STM_MAX_CHARS } from "./constants"

export interface StmEntry {
	role: "user" | "assistant"
	content: string
	timestamp: number
}

export class ShortTermMemory {
	private entries: StmEntry[] = []
	private totalChars = 0

	constructor(private readonly maxChars: number = STM_MAX_CHARS) {}

	/** Append a new step to the STM. Trims head if over budget. */
	push(role: StmEntry["role"], content: string): void {
		const entry: StmEntry = { role, content, timestamp: Date.now() }
		this.entries.push(entry)
		this.totalChars += content.length

		// Trim oldest entries until within budget
		while (this.totalChars > this.maxChars && this.entries.length > 1) {
			const evicted = this.entries.shift()!
			this.totalChars -= evicted.content.length
		}
	}

	/** Return all current entries (oldest first). */
	getEntries(): readonly StmEntry[] {
		return this.entries
	}

	/**
	 * Produce a compact text summary of the STM for use in afterRun().
	 * Format: role: content\n...
	 * Capped at maxChars characters.
	 */
	summarize(): string {
		return this.entries.map((e) => `${e.role}: ${e.content}`).join("\n")
	}

	get charCount(): number {
		return this.totalChars
	}

	clear(): void {
		this.entries = []
		this.totalChars = 0
	}
}
