/**
 * EpisodicMemoryService
 *
 * Implements:
 *  - Persistent episodic memory (dual-write: .njust_ai + .roo)
 *  - Two-Phase Retrieval (Phase A: cosine sim threshold, Phase B: z-score composite)
 *  - Monte Carlo Q-value update
 */

import * as fs from "fs/promises"
import * as path from "path"
import { safeWriteJson } from "../../../utils/safeWriteJson"
import { MemoryEmbeddingAdapter } from "./MemoryEmbeddingAdapter"
import {
	ALPHA,
	EPISODIC_FILE,
	LAMBDA,
	LTM_DISTILL_INTERVAL,
	MEMRL_PRIMARY_DIR,
	MEMRL_ROO_DIR,
	Q_INIT,
	SIM_THRESHOLD,
	TOP_K1,
	TOP_K2,
} from "./constants"

export interface EpisodicEntry {
	id: string
	/** Plain-text intent / task description */
	intent: string
	/** Embedding vector for this intent */
	embedding: number[]
	/** Short-term memory summary from the task run */
	stmSummary: string
	/** Monte Carlo Q-value (expected reward) */
	qValue: number
	/** Number of times this entry has been updated */
	updateCount: number
	createdAt: number
	updatedAt: number
}

export interface EpisodicStore {
	entries: EpisodicEntry[]
	/** Total number of episodic writes ever (used to trigger LTM distillation) */
	totalWrites: number
}

export class EpisodicMemoryService {
	private store: EpisodicStore = { entries: [], totalWrites: 0 }
	private loaded = false

	constructor(
		private readonly workspaceDir: string,
		private readonly embedder: MemoryEmbeddingAdapter,
		/** Called when totalWrites % LTM_DISTILL_INTERVAL === 0 */
		private readonly onDistillTrigger?: () => void,
	) {}

	// ── Persistence ────────────────────────────────────────────────────────────

	private get primaryPath(): string {
		return path.join(this.workspaceDir, MEMRL_PRIMARY_DIR, EPISODIC_FILE)
	}

	private get rooPath(): string {
		return path.join(this.workspaceDir, MEMRL_ROO_DIR, EPISODIC_FILE)
	}

	async load(): Promise<void> {
		if (this.loaded) return
		try {
			const raw = await fs.readFile(this.primaryPath, "utf-8")
			this.store = JSON.parse(raw) as EpisodicStore
		} catch {
			// Cold start — empty store is fine
			this.store = { entries: [], totalWrites: 0 }
		}
		this.loaded = true
	}

	private async persist(): Promise<void> {
		await safeWriteJson(this.primaryPath, this.store)
		// Dual-write mirror for acceptance criteria
		await safeWriteJson(this.rooPath, this.store).catch(() => {
			/* best-effort */
		})
	}

	// ── Write ──────────────────────────────────────────────────────────────────

	/**
	 * Store a new episodic entry and update totalWrites.
	 * Triggers LTM distillation callback every LTM_DISTILL_INTERVAL writes.
	 */
	async write(intent: string, stmSummary: string, reward: number): Promise<void> {
		await this.load()

		const embedding = await this.embedder.embed(intent)
		const now = Date.now()
		const id = `ep_${now}_${Math.random().toString(36).slice(2, 9)}`

		const entry: EpisodicEntry = {
			id,
			intent,
			embedding,
			stmSummary,
			qValue: Q_INIT + ALPHA * (reward - Q_INIT),
			updateCount: 1,
			createdAt: now,
			updatedAt: now,
		}

		this.store.entries.push(entry)
		this.store.totalWrites++

		await this.persist()

		if (this.store.totalWrites % LTM_DISTILL_INTERVAL === 0) {
			this.onDistillTrigger?.()
		}
	}

	/**
	 * Update Q-value for an existing entry via Monte Carlo rule:
	 *   Q_new = Q_old + α·(r - Q_old)
	 */
	async updateQ(entryId: string, reward: number): Promise<void> {
		await this.load()

		const entry = this.store.entries.find((e) => e.id === entryId)
		if (!entry) return

		entry.qValue = entry.qValue + ALPHA * (reward - entry.qValue)
		entry.updateCount++
		entry.updatedAt = Date.now()

		await this.persist()
	}

	// ── Two-Phase Retrieval ────────────────────────────────────────────────────

	/**
	 * Retrieve the most relevant episodic entries for a given query intent.
	 *
	 * Phase A: cosine similarity ≥ SIM_THRESHOLD → candidate set (up to TOP_K1)
	 * Phase B: z-score normalise sim and Q, then score = (1-λ)·sim̂ + λ·Q̂ → top TOP_K2
	 */
	async retrieve(queryIntent: string): Promise<EpisodicEntry[]> {
		await this.load()

		if (this.store.entries.length === 0) return []

		const queryVec = await this.embedder.embed(queryIntent)
		if (queryVec.length === 0) return []

		// ── Phase A ──────────────────────────────────────────────────────────
		const withSim = this.store.entries
			.map((entry) => ({
				entry,
				sim: MemoryEmbeddingAdapter.cosineSim(queryVec, entry.embedding),
			}))
			.filter((x) => x.sim >= SIM_THRESHOLD)
			.sort((a, b) => b.sim - a.sim)
			.slice(0, TOP_K1)

		if (withSim.length === 0) return []

		// ── Phase B ──────────────────────────────────────────────────────────
		const sims = withSim.map((x) => x.sim)
		const qVals = withSim.map((x) => x.entry.qValue)

		const simHat = MemoryEmbeddingAdapter.zScore(sims)
		const qHat = MemoryEmbeddingAdapter.zScore(qVals)

		const scored = withSim.map((x, i) => ({
			entry: x.entry,
			score: (1 - LAMBDA) * (simHat[i] ?? 0) + LAMBDA * (qHat[i] ?? 0),
		}))

		scored.sort((a, b) => b.score - a.score)

		return scored.slice(0, TOP_K2).map((x) => x.entry)
	}

	/** Return up to `n` most-recent entries (for LTM distillation). */
	getRecent(n: number): EpisodicEntry[] {
		return [...this.store.entries].sort((a, b) => b.createdAt - a.createdAt).slice(0, n)
	}

	get totalWrites(): number {
		return this.store.totalWrites
	}
}
