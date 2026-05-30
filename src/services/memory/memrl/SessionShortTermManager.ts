/**
 * SessionShortTermManager
 *
 * Manages a Map<taskId, ShortTermMemory> with LRU eviction.
 * Cap: STM_LRU_LIMIT concurrent entries.
 */

import { STM_LRU_LIMIT, STM_MAX_CHARS } from "./constants"
import { ShortTermMemory } from "./ShortTermMemory"

export class SessionShortTermManager {
	/** Insertion-order map for LRU (Map preserves insertion order). */
	private readonly store = new Map<string, ShortTermMemory>()

	constructor(
		private readonly maxEntries: number = STM_LRU_LIMIT,
		private readonly maxCharsPerTask: number = STM_MAX_CHARS,
	) {}

	/** Get or create a ShortTermMemory for the given taskId. */
	get(taskId: string): ShortTermMemory {
		if (this.store.has(taskId)) {
			// Move to end (most recently used)
			const stm = this.store.get(taskId)!
			this.store.delete(taskId)
			this.store.set(taskId, stm)
			return stm
		}

		// Evict LRU if at capacity
		if (this.store.size >= this.maxEntries) {
			const lruKey = this.store.keys().next().value as string
			this.store.delete(lruKey)
		}

		const stm = new ShortTermMemory(this.maxCharsPerTask)
		this.store.set(taskId, stm)
		return stm
	}

	/** Delete the STM for a finished task (free memory). */
	delete(taskId: string): void {
		this.store.delete(taskId)
	}

	get size(): number {
		return this.store.size
	}
}
