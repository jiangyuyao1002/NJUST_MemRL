import * as vscode from "vscode"
import pWaitFor from "p-wait-for"
import crypto from "crypto"

import {
	type ContextCondense,
	TelemetryEventName,
} from "@njust-ai/types"
import { Package } from "../../shared/package"
import { defaultModeSlug } from "../../shared/modes"

import type { ApiHandlerCreateMessageMetadata } from "../../api"
import { resolveParallelNativeToolCalls } from "../../shared/parallelToolCalls"

import type { IMcpHubService } from "../../services/mcp/interfaces/IMcpHubService"
import { McpServerManager } from "../../services/mcp/McpServerManager"

import {
	SYSTEM_PROMPT_PARTS,
	type SystemPromptParts,
	deriveCangjieContextTokenBudgetFromContextWindow,
} from "../prompts/system"
import { getCangjieSystemPromptCacheKeySuffix } from "../prompts/sections/cangjie-context"
import type { ApiMessage } from "../task-persistence"
import { buildNativeToolsArrayWithRestrictions } from "./build-tools"
import { getEnvironmentDetails } from "../environment/getEnvironmentDetails"
import { summarizeConversation } from "../condense"

import type { Task } from "./Task"
import { logger } from "../../shared/logger"
import { TelemetryService } from "@njust-ai/telemetry"

/**
 * TaskRequestBuilder handles system prompt generation, prompt caching,
 * and context condensation logic extracted from Task.ts.
 *
 * Uses the delegation pattern: accesses Task instance properties via `this.task`.
 */
export class TaskRequestBuilder {
	private systemPromptPartsCache?: { key: string; parts: SystemPromptParts; time: number }

	/** In-flight prefetch promise to avoid duplicate concurrent prefetches. */
	private prefetchInFlight?: Promise<void>

	constructor(private task: Task) {}

	/** Last user message text for Ask/Architect 仓颉语料相关性检测（与缓存键一致）。 */
	private getLastUserMessageTextForCangjieHint(): string | undefined {
		const history = this.task.apiConversationHistory
		for (let i = history.length - 1; i >= 0; i--) {
			const m = history[i] as ApiMessage
			if (m.role !== "user") continue
			const c = m.content
			if (typeof c === "string") return c
			if (Array.isArray(c)) {
				const parts: string[] = []
				for (const block of c) {
					if (block && typeof block === "object" && (block as { type?: string }).type === "text") {
						const t = (block as { text?: string }).text
						if (typeof t === "string") parts.push(t)
					}
				}
				if (parts.length > 0) return parts.join("\n")
			}
		}
		return undefined
	}

	/**
	 * Inherit system prompt parts cache from a parent task's builder.
	 */
	inheritCacheFromParent(parentTask: Task): void {
		const parentCache = parentTask.requestBuilder?.systemPromptPartsCache
		if (!parentCache) return

		// Skip cache inheritance if parent and child modes are known to differ.
		// Different modes produce different system prompts, making parent cache invalid.
		const parentMode = parentTask.taskMode
		const childMode = this.task.taskMode
		if (parentMode !== undefined && childMode !== undefined && parentMode !== childMode) return

		this.systemPromptPartsCache = {
			key: parentCache.key,
			parts: parentCache.parts,
			time: Date.now(),
		}
		if (parentTask.cachedStreamingModel) {
			this.task.cachedStreamingModel = parentTask.cachedStreamingModel
		}

		// Inherit cached tool definitions so child tasks skip redundant tool building.
		if (parentTask.cachedToolDefinitions) {
			this.task.cachedToolDefinitions = { ...parentTask.cachedToolDefinitions, time: Date.now() }
		}
	}

	/**
	 * Generate system prompt parts with caching support.
	 */
	async getSystemPromptParts(): Promise<SystemPromptParts> {
		const { mcpEnabled } = (await this.task.providerRef.deref()?.getState()) ?? {}
		let mcpHub: IMcpHubService | undefined
		if (mcpEnabled ?? true) {
			const provider = this.task.providerRef.deref()

			if (!provider) {
				throw new Error("Provider reference lost during view transition")
			}

			// Wait for MCP hub initialization through McpServerManager
			mcpHub = await McpServerManager.getInstance(provider.context, provider)

			if (!mcpHub) {
				throw new Error("Failed to get MCP hub from server manager")
			}

			// Wait for MCP servers to be connected before generating system prompt
			await pWaitFor(() => !mcpHub!.isConnecting, { timeout: 10_000 }).catch(() => {
				logger.error("TaskRequestBuilder", "MCP servers failed to connect in time")
			})
		}

		const rooIgnoreInstructions = this.task.rooIgnoreController?.getInstructions()

		const state = await this.task.providerRef.deref()?.getState()

		const {
			mode,
			customModes,
			customModePrompts,
			customInstructions,
			experiments,
			language,
			apiConfiguration,
			enableSubfolderRules,
			enableWebSearch,
		} = state ?? {}

		return await (async () => {
			const mcpServers = mcpHub?.getServers() ?? []
			const mcpToolNames = mcpServers
				.flatMap((server) =>
					(server.tools ?? []).map((tool) => `${server.name}:${tool.name}`),
				)
				.sort()
			const mcpToolsHash = crypto.createHash("sha256").update(mcpToolNames.join("|")).digest("hex").slice(0, 12)
			const instructionHash = crypto
				.createHash("sha256")
				.update(`${customInstructions ?? ""}|${JSON.stringify(customModePrompts ?? {})}`)
				.digest("hex")
				.slice(0, 12)
			const lastUserForCangjie = this.getLastUserMessageTextForCangjieHint()
			const staticKey = JSON.stringify({
				cwd: this.task.cwd,
				mode,
				lang: language,
				enableWebSearch,
				useAgentRules: vscode.workspace.getConfiguration(Package.name).get<boolean>("useAgentRules") ?? true,
				enableSubfolderRules: enableSubfolderRules ?? false,
				mcpEnabled: Boolean(mcpHub),
				mcpServerCount: mcpServers.length,
				mcpToolsHash,
				instructionHash,
				cangjieAugmentKey: getCangjieSystemPromptCacheKeySuffix(
					this.task.cwd,
					mode ?? defaultModeSlug,
					lastUserForCangjie,
				),
			})
			const cacheKey = `${mode ?? defaultModeSlug}:${staticKey}`
			const now = Date.now()
			if (this.systemPromptPartsCache && this.systemPromptPartsCache.key === cacheKey && now - this.systemPromptPartsCache.time < 30_000) {
				return this.systemPromptPartsCache.parts
			}
			const provider = this.task.providerRef.deref()

			if (!provider) {
				throw new Error("Provider not available")
			}

			const modelInfo = this.task.api.getModel().info

			const parts = await SYSTEM_PROMPT_PARTS(
				provider.context,
				this.task.cwd,
				false,
				mcpHub,
				this.task.diffStrategy,
				mode ?? defaultModeSlug,
				customModePrompts,
				customModes,
				customInstructions,
				experiments,
				language,
				rooIgnoreInstructions,
				{
					todoListEnabled: apiConfiguration?.todoListEnabled ?? true,
					useAgentRules:
						vscode.workspace.getConfiguration(Package.name).get<boolean>("useAgentRules") ?? true,
					enableSubfolderRules: enableSubfolderRules ?? false,
					newTaskRequireTodos: vscode.workspace
						.getConfiguration(Package.name)
						.get<boolean>("newTaskRequireTodos", false),
					isStealthModel: modelInfo?.isStealthModel,
					enableWebSearch: enableWebSearch ?? false,
					cangjieContextTokenBudget: deriveCangjieContextTokenBudgetFromContextWindow(
						modelInfo?.contextWindow,
					),
					cangjieContextIntensity: this.task.cangjieRuntimePolicy.getContextIntensity(
						Math.max(0, this.task.apiConversationHistory.length - 1),
					),
					cangjieRecentBuildRootCauses: this.task.cangjieRuntimePolicy.getRecentBuildRootCauses(),
					cangjieRepairDirective: this.task.cangjieRuntimePolicy.getRepairDirective(),
					contextWindow: modelInfo?.contextWindow,
					taskId: this.task.taskId,
					turnIndex: Math.max(0, this.task.apiConversationHistory.length - 1),
					enableTurnAwarePromptPruning: (state as Record<string, UnsafeAny>)?.enableTurnAwarePromptPruning ?? true,
					lastUserMessageForCangjieHint: lastUserForCangjie,
					memrlEpisodicHints: this.task.memrlEpisodicHints,
					memrlLtmRules: this.task.memrlLtmRules,
				},
				undefined, // todoList
				this.task.api.getModel().id,
				provider.getSkillsManager(),
			)
			this.systemPromptPartsCache = { key: cacheKey, parts, time: now }
			return parts
		})()
	}

	/**
	 * Get the full system prompt string.
	 */
	async getSystemPrompt(): Promise<string> {
		const parts = await this.getSystemPromptParts()
		return parts.fullPrompt
	}

	/**
	 * Condense the conversation context to reduce token usage.
	 */
	async condenseContext(): Promise<void> {
		// CRITICAL: Flush any pending tool results before condensing
		// to ensure tool_use/tool_result pairs are complete in history
		await this.task.flushPendingToolResultsToHistory()

		const systemPrompt = await this.getSystemPrompt()

		// Get condensing configuration
		const state = await this.task.providerRef.deref()?.getState()
		this.systemPromptPartsCache = undefined
		const customCondensingPrompt = state?.customSupportPrompts?.CONDENSE
		const { mode, apiConfiguration } = state ?? {}

		const { contextTokens: prevContextTokens } = this.task.getTokenUsage()

		// Build tools for condensing metadata (same tools used for normal API calls)
		const provider = this.task.providerRef.deref()
		let allTools: import("openai").default.Chat.ChatCompletionTool[] = []
		if (provider) {
			const modelInfo = this.task.api.getModel().info
			const toolsResult = await buildNativeToolsArrayWithRestrictions({
				provider,
				cwd: this.task.cwd,
				mode,
				customModes: state?.customModes,
				experiments: state?.experiments,
				apiConfiguration,
				disabledTools: state?.disabledTools,
				enableWebSearch: state?.enableWebSearch,
				modelInfo,
				includeAllToolsWithRestrictions: false,
			})
			allTools = toolsResult.tools
		}

		// Build metadata with tools and taskId for the condensing API call
		const metadata: ApiHandlerCreateMessageMetadata = {
			mode,
			taskId: this.task.taskId,
			...(allTools.length > 0
				? {
						tools: allTools,
						tool_choice: "auto",
						parallelToolCalls: resolveParallelNativeToolCalls(apiConfiguration),
					}
				: {}),
		}
		// Generate environment details to include in the condensed summary
		const environmentDetails = await getEnvironmentDetails(this.task, true)

		const filesReadByRoo = await this.getFilesReadByRooSafely("condenseContext")

		const {
			messages,
			summary,
			cost,
			newContextTokens = 0,
			error,
			condenseId,
		} = await summarizeConversation({
			messages: this.task.apiConversationHistory,
			apiHandler: this.task.api,
			systemPrompt,
			taskId: this.task.taskId,
			isAutomaticTrigger: false,
			customCondensingPrompt,
			metadata,
			environmentDetails,
			filesReadByRoo,
			cwd: this.task.cwd,
			rooIgnoreController: this.task.rooIgnoreController,
		})
		if (error) {
			await this.task.say(
				"condense_context_error",
				error,
				undefined /* images */,
				false /* partial */,
				undefined /* checkpoint */,
				undefined /* progressStatus */,
				{ isNonInteractive: true } /* options */,
			)
			return
		}
		await this.task.overwriteApiConversationHistory(messages)

		const contextCondense: ContextCondense = {
			summary,
			cost,
			newContextTokens,
			prevContextTokens,
			condenseId: condenseId!,
		}
		await this.task.say(
			"condense_context",
			undefined /* text */,
			undefined /* images */,
			false /* partial */,
			undefined /* checkpoint */,
			undefined /* progressStatus */,
			{ isNonInteractive: true } /* options */,
			contextCondense,
		)

		// Process any queued messages after condensing completes
		this.task.processQueuedMessages()
	}

	/**
	 * Prefetch system prompt data in the background.
	 * Triggers `getSystemPromptParts()` which populates the 30-second cache.
	 * Uses `Promise.allSettled` so individual failures don't block others.
	 * Returns immediately if a valid cache exists or a prefetch is already in flight.
	 */
	async prefetchSystemPromptData(): Promise<void> {
		// Skip if we already have a valid cache (< 30s old)
		if (this.systemPromptPartsCache && Date.now() - this.systemPromptPartsCache.time < 30_000) {
			return
		}

		// Deduplicate: if a prefetch is already running, wait for it instead of starting another.
		if (this.prefetchInFlight) {
			return this.prefetchInFlight
		}

		const startTime = Date.now()
		this.prefetchInFlight = (async () => {
			try {
				// getSystemPromptParts() internally calls SYSTEM_PROMPT_PARTS which
				// runs I/O-heavy operations (skills discovery, Cangjie context,
				// rules loading, modes section, custom instructions) and caches
				// the result for 30 seconds.
				const results = await Promise.allSettled([
					this.getSystemPromptParts(),
				])

				const elapsed = Date.now() - startTime
				const failures = results.filter((r) => r.status === "rejected")
				if (failures.length > 0) {
			logger.warn("TaskRequestBuilder",
					`System prompt data prefetched in ${elapsed}ms with ${failures.length} failure(s)`,
					failures.map((f) => (f as PromiseRejectedResult).reason),
				)
				} else {
					logger.info("TaskRequestBuilder", `System prompt data prefetched in ${elapsed}ms`)
				}
			} catch (error) {
				logger.warn("TaskRequestBuilder", "System prompt prefetch failed:", error)
				TelemetryService.reportError(error instanceof Error ? error : new Error(String(error)), TelemetryEventName.UTILITY_ERROR)
			}
		})().finally(() => {
			this.prefetchInFlight = undefined
		})

		return this.prefetchInFlight
	}

	/**
	 * Clear the system prompt parts cache.
	 */
	clearCache(): void {
		this.systemPromptPartsCache = undefined
	}

	/**
	 * Safely get files read by Njust-AI, catching errors.
	 */
	private async getFilesReadByRooSafely(_context: string): Promise<string[] | undefined> {
		try {
			return await this.task.fileContextTracker.getFilesReadByRoo()
		} catch (error) {
			logger.error("TaskRequestBuilder", `Failed to get files read by Njust-AI:`, error)
			TelemetryService.reportError(error instanceof Error ? error : new Error(String(error)), TelemetryEventName.UTILITY_ERROR)
			return undefined
		}
	}
}
