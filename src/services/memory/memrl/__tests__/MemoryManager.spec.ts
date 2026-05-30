/**
 * Integration test: beforeRun → prompt hints → afterRun → Q update flow.
 */
import { describe, it, expect, vi, beforeEach } from "vitest"
import { MemoryManager } from "../MemoryManager"
import type { IEmbedder } from "../../../code-index/interfaces/embedder"
import type { ApiHandler } from "../../../../api"

vi.mock("../../../../utils/safeWriteJson", () => ({
	safeWriteJson: vi.fn().mockResolvedValue(undefined),
}))

function makeEmbedder(): IEmbedder {
	return {
		createEmbeddings: vi.fn().mockResolvedValue({ embeddings: [[0.6, 0.8]] }),
		validateConfiguration: vi.fn().mockResolvedValue({ valid: true }),
		get embedderInfo() {
			return { name: "openai" as const }
		},
	}
}

function makeApi(): ApiHandler {
	return {
		createMessage: vi.fn().mockReturnValue((async function* () {})()),
		getModel: vi.fn().mockReturnValue({ id: "gpt-4o", info: {} }),
	} as unknown as ApiHandler
}

describe("MemoryManager integration", () => {
	let mgr: MemoryManager

	beforeEach(() => {
		mgr = new MemoryManager("/tmp/memrl-mgr-test")
		mgr.updateDependencies(makeApi(), makeEmbedder())
	})

	it("beforeRun returns empty strings on cold start", async () => {
		const result = await mgr.beforeRun("task-1", "build the feature")
		expect(result.episodicHints).toBe("")
		expect(result.ltmRules).toBe("")
	})

	it("afterRun does not throw and is fire-and-forget", () => {
		expect(() => mgr.afterRun("task-1", "build the feature", "summary text", 1.0)).not.toThrow()
	})

	it("getStm returns a ShortTermMemory", () => {
		const stm = mgr.getStm("task-1")
		expect(stm).toBeDefined()
		stm.push("user", "hello")
		expect(stm.getEntries()).toHaveLength(1)
	})

	it("returns empty results if updateDependencies was not called", async () => {
		const fresh = new MemoryManager("/tmp/memrl-fresh")
		const result = await fresh.beforeRun("task-x", "intent")
		expect(result.episodicHints).toBe("")
		expect(result.ltmRules).toBe("")
	})

	it("full flow: write then retrieve produces formatted hints", async () => {
		// Write a high-reward episode
		await (mgr as any).episodic?.write("implement login feature", "user: implement\nassistant: done", 1.0)
		// Now retrieve
		const result = await mgr.beforeRun("task-2", "implement login feature")
		// Should have episodic hints since we wrote with same vector
		expect(result.episodicHints).toContain("Past Episodes")
	})
})
