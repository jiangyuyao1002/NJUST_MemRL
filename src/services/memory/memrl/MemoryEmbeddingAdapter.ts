/**
 * MemoryEmbeddingAdapter
 *
 * Bridges the existing IEmbedder interface for MemRL usage.
 * Also exposes static math utilities (cosineSim, zScore) used
 * by Phase A/B retrieval.
 */

import type { IEmbedder } from "../../code-index/interfaces/embedder"

export class MemoryEmbeddingAdapter {
	constructor(private readonly embedder: IEmbedder) {}

	/**
	 * Embed a single text string.
	 * Returns a float32 vector or empty array on failure.
	 */
	async embed(text: string): Promise<number[]> {
		try {
			const resp = await this.embedder.createEmbeddings([text])
			return resp.embeddings[0] ?? []
		} catch {
			return []
		}
	}

	/**
	 * Embed multiple texts in one batch call.
	 */
	async embedBatch(texts: string[]): Promise<number[][]> {
		if (texts.length === 0) return []
		try {
			const resp = await this.embedder.createEmbeddings(texts)
			return resp.embeddings
		} catch {
			return texts.map(() => [])
		}
	}

	// ── Static math helpers ────────────────────────────────────────────────────

	/**
	 * Cosine similarity ∈ [-1, 1].
	 * Returns 0 if either vector is zero-length.
	 */
	static cosineSim(a: number[], b: number[]): number {
		if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0
		let dot = 0
		let normA = 0
		let normB = 0
		for (let i = 0; i < a.length; i++) {
			const ai = a[i] ?? 0
			const bi = b[i] ?? 0
			dot += ai * bi
			normA += ai * ai
			normB += bi * bi
		}
		const denom = Math.sqrt(normA) * Math.sqrt(normB)
		return denom === 0 ? 0 : dot / denom
	}

	/**
	 * Z-score normalise an array in-place (mutates a copy).
	 * Returns the standardised array (mean=0, std=1).
	 * If std=0, all values map to 0.
	 */
	static zScore(values: number[]): number[] {
		if (values.length === 0) return []
		const mean = values.reduce((s, v) => s + v, 0) / values.length
		const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length
		const std = Math.sqrt(variance)
		if (std === 0) return values.map(() => 0)
		return values.map((v) => (v - mean) / std)
	}
}
