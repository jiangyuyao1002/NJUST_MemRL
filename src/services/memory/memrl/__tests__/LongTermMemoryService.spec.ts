import { describe, it, expect, vi, beforeEach } from "vitest"
import { LongTermMemoryService } from "../LongTermMemoryService"
import { MemoryEmbeddingAdapter } from "../MemoryEmbeddingAdapter"
import type { IEmbedder } from "../../../code-index/interfaces/embedder"
import type { ApiHandler } from "../../../../api"

vi.mock("../../../../utils/safeWriteJson", () => ({
	safeWriteJson: vi.fn().mockResolvedValue(undefined),
}))

function makeEmbedder(): MemoryEmbeddingAdapter {
	const e: IEmbedder = {
		createEmbeddings: vi.fn().mockResolvedValue({ embeddings: [[0.6, 0.8]] }),
		validateConfiguration: vi.fn().mockResolvedValue({ valid: true }),
		get embedderInfo() {
			return { name: "openai" as const }
		},
	}
	return new MemoryEmbeddingAdapter(e)
}

function makeApi(responseJson: string): ApiHandler {
	const stream = (async function* () {
		yield { type: "text" as const, text: responseJson }
	})()
	return {
		createMessage: vi.fn().mockReturnValue(stream),
		getModel: vi.fn().mockReturnValue({ id: "gpt-4o", info: {} }),
	} as unknown as ApiHandler
}

describe("LongTermMemoryService", () => {
	let svc: LongTermMemoryService

	beforeEach(() => {
		const ruleJson = JSON.stringify([
			{
				topic: "Testing strategy",
				rule: "Always write unit tests first.",
				examples: ["write test then code"],
				confidence: 0.9,
			},
		])
		svc = new LongTermMemoryService("/tmp/ltm-test", makeEmbedder(), makeApi(ruleJson))
	})

	it("starts with no rules", () => {
		expect(svc.getRules()).toHaveLength(0)
	})

	it("distill adds rules from LLM response", async () => {
		await svc.distill([
			{
				id: "ep1",
				intent: "write tests",
				embedding: [0.6, 0.8],
				stmSummary: "wrote tests",
				qValue: 0.9,
				updateCount: 1,
				createdAt: Date.now(),
				updatedAt: Date.now(),
			},
		])
		expect(svc.getRules().length).toBeGreaterThan(0)
		expect(svc.getRules()[0].topic).toBe("Testing strategy")
	})

	it("retrieve returns rules above threshold", async () => {
		await svc.distill([
			{
				id: "ep1",
				intent: "write tests",
				embedding: [0.6, 0.8],
				stmSummary: "summary",
				qValue: 0.9,
				updateCount: 1,
				createdAt: Date.now(),
				updatedAt: Date.now(),
			},
		])
		const results = await svc.retrieve("write tests")
		expect(results.length).toBeGreaterThan(0)
	})

	it("markUsed increments useCount", async () => {
		await svc.distill([
			{
				id: "ep1",
				intent: "test",
				embedding: [0.6, 0.8],
				stmSummary: "s",
				qValue: 0.9,
				updateCount: 1,
				createdAt: Date.now(),
				updatedAt: Date.now(),
			},
		])
		const rules = svc.getRules()
		expect(rules.length).toBeGreaterThan(0)
		const ruleId = rules[0].id
		await svc.markUsed([ruleId])
		expect(svc.getRules().find((r) => r.id === ruleId)?.useCount).toBe(1)
	})
})
