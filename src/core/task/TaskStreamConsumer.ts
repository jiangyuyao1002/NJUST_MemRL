import type { Anthropic } from "@anthropic-ai/sdk"
import type { ClineApiReqCancelReason, ClineMessage } from "@njust-ai/types"
import {
	clineApiReqInfoSchema,
	DEFAULT_AUTO_CONDENSE_CONTEXT_PERCENT,
	getApiProtocol,
	getModelId,
	isRetiredProvider,
	TelemetryEventName,
} from "@njust-ai/types"
import pWaitFor from "p-wait-for"

import type { TaskExecutorHost } from "./interfaces/ITaskExecutorHost"
import type { ApiStream, GroundingSource } from "../../api/transform/stream"
import type { ToolUse, McpToolUse } from "../../shared/tools"
import type { TypedBlock } from "../assistant-message/types"

import { NativeToolCallParser } from "../assistant-message/NativeToolCallParser"
import { processTaskStreamChunk, finalizePendingStreamingToolCalls } from "./TaskStreamChunkProcessor"
import { TaskState } from "./TaskStateMachine"
import { TaskAbortedError } from "./TaskErrors"
import { handleMidStreamFailure, handleEmptyAssistantResponse } from "./TaskRetryHandler"

import { findLastIndex } from "../../shared/array"
import { calculateApiCostAnthropic, calculateApiCostOpenAI } from "../../shared/cost"
import { markUserContentReadyIfDrained } from "../assistant-message/streamState"
import { isAnyToolUse, isToolUseBlock } from "../assistant-message/types"
import { formatResponse } from "../prompts/responses"
import { willManageContext } from "../context-management"
import { globalQueryProfiler } from "../../utils/queryProfiler"
import { TelemetryService } from "@njust-ai/telemetry"
import { globalCacheMetrics } from "../../utils/cacheMetrics"
import { globalPromptCacheBreakDetector } from "../prompts/promptCacheBreakDetection"
import { sanitizeToolUseId } from "../../utils/tool-id"
import { logger } from "../../shared/logger"
import { getErrorMessage } from "../../shared/error-utils"
import { debugLog } from "../../utils/debugLog"
import { t as i18nT } from "../../i18n"

const DEFAULT_USAGE_COLLECTION_TIMEOUT_MS = 5000
const USER_MESSAGE_CONTENT_READY_TIMEOUT_MS = 30_000

export interface StackItem {
	userContent: Anthropic.Messages.ContentBlockParam[]
	includeFileDetails: boolean
	retryAttempt?: number
	userMessageWasRemoved?: boolean
}

export type FinalizeToolUseFn = (
	task: TaskExecutorHost,
	id: string,
	finalToolUse: ToolUse | McpToolUse,
) => ToolUse | McpToolUse

export interface StreamConsumptionResult {
	assistantMessage: string
	reasoningMessage: string
	pendingGroundingSources: GroundingSource[]
	action: "proceed" | "continue" | "break"
}

export interface ConsumeStreamConfig {
	task: TaskExecutorHost
	stream: ApiStream
	toolCallParser: NativeToolCallParser
	placeFinalizedStreamingToolUse: FinalizeToolUseFn
	requestProfileId: string
	lastApiReqIndex: number
	requestStartedAt: number
	retryAttempt: number
	currentUserContent: Anthropic.Messages.ContentBlockParam[]
	stack: StackItem[]
}

export async function consumeApiStream(config: ConsumeStreamConfig): Promise<StreamConsumptionResult> {
	const {
		task: t,
		stream,
		toolCallParser,
		placeFinalizedStreamingToolUse,
		requestProfileId,
		lastApiReqIndex,
		requestStartedAt,
		retryAttempt,
		currentUserContent,
		stack,
	} = config

	let cacheWriteTokens = 0
	let cacheReadTokens = 0
	let inputTokens = 0
	let outputTokens = 0
	let totalCost: number | undefined
	let assistantMessage = ""
	let reasoningMessage = ""
	const pendingGroundingSources: GroundingSource[] = []

	const streamModelInfo = t.cachedStreamingModel!.info
	const cachedModelId = t.cachedStreamingModel!.id

	const updateApiReqMsg = (cancelReason?: ClineApiReqCancelReason, streamingFailedMessage?: string) => {
		if (lastApiReqIndex < 0 || !t.clineMessages[lastApiReqIndex]) {
			return
		}

		const existingData = clineApiReqInfoSchema.parse(JSON.parse(t.clineMessages[lastApiReqIndex].text || "{}"))

		const modelId = getModelId(t.apiConfiguration)
		const apiProvider = t.apiConfiguration.apiProvider
		const apiProtocol = getApiProtocol(
			apiProvider && !isRetiredProvider(apiProvider) ? apiProvider : undefined,
			modelId,
		)

		const costResult =
			apiProtocol === "anthropic"
				? calculateApiCostAnthropic(
						streamModelInfo,
						inputTokens,
						outputTokens,
						cacheWriteTokens,
						cacheReadTokens,
					)
				: calculateApiCostOpenAI(streamModelInfo, inputTokens, outputTokens, cacheWriteTokens, cacheReadTokens)

		t.clineMessages[lastApiReqIndex].text = JSON.stringify({
			...existingData,
			tokensIn: costResult.totalInputTokens,
			tokensOut: costResult.totalOutputTokens,
			cacheWrites: cacheWriteTokens,
			cacheReads: cacheReadTokens,
			cost: totalCost ?? costResult.totalCost,
			cancelReason,
			streamingFailedMessage,
		} satisfies import("@njust-ai/types").ClineApiReqInfo)
	}

	const abortStream = async (cancelReason: ClineApiReqCancelReason, streamingFailedMessage?: string) => {
		if (t.diffViewProvider.isEditing) {
			await t.diffViewProvider.revertChanges()
		}

		const lastMessage = t.clineMessages.at(-1)

		if (lastMessage?.partial) {
			lastMessage.partial = false
		}

		updateApiReqMsg(cancelReason, streamingFailedMessage)
		await t.saveClineMessages()

		t.didFinishAbortingStream = true
	}

	let streamAction: "proceed" | "continue" | "break" = "proceed"

	try {
		const iterator = stream[Symbol.asyncIterator]()

		const nextChunkWithAbort = async () => {
			const nextPromise = iterator.next()

			if (t.currentRequestAbortController) {
				const signal = t.currentRequestAbortController!.signal
				let onAbort: (() => void) | undefined
				const abortPromise = new Promise<never>((_, reject) => {
					if (signal.aborted) {
						reject(new Error("Request cancelled by user"))
						return
					}
					onAbort = () => reject(new Error("Request cancelled by user"))
					signal.addEventListener("abort", onAbort, { once: true })
				})
				try {
					return await Promise.race([nextPromise, abortPromise])
				} finally {
					if (onAbort) signal.removeEventListener("abort", onAbort)
				}
			}

			return await nextPromise
		}

		let item = await nextChunkWithAbort()
		while (!item.done) {
			const chunk = item.value
			item = await nextChunkWithAbort()
			if (!chunk) {
				continue
			}

			await processTaskStreamChunk({
				task: t,
				chunk,
				toolCallParser,
				requestProfileId,
				pendingGroundingSources,
				finalizeToolUse: placeFinalizedStreamingToolUse,
				appendReasoningText: (text) => {
					reasoningMessage += text
					return reasoningMessage
				},
				appendAssistantText: (text) => {
					assistantMessage += text
					return assistantMessage
				},
				addUsage: (usageChunk) => {
					inputTokens += usageChunk.inputTokens
					outputTokens += usageChunk.outputTokens
					cacheWriteTokens += usageChunk.cacheWriteTokens ?? 0
					cacheReadTokens += usageChunk.cacheReadTokens ?? 0
					totalCost = usageChunk.totalCost
				},
			})
			if (t.abort) {
				logger.info("TaskStreamConsumer", `Aborting stream for task ${t.taskId}, abandoned = ${t.abandoned}`)

				if (!t.abandoned) {
					await abortStream("user_cancelled")
				}

				break
			}

			if (t.didRejectTool) {
				assistantMessage += "\n\n[Response interrupted by user feedback]"
				break
			}

			if (t.didAlreadyUseTool) {
				assistantMessage +=
					"\n\n[Response interrupted by a tool use result. Only one tool may be used at a time and should be placed at the end of the message.]"
				break
			}
		}

		const currentTokens = {
			input: inputTokens,
			output: outputTokens,
			cacheWrite: cacheWriteTokens,
			cacheRead: cacheReadTokens,
			total: totalCost,
		}

		const drainStreamInBackgroundToFindAllUsage = async (apiReqIndex: number) => {
			const timeoutMs = DEFAULT_USAGE_COLLECTION_TIMEOUT_MS
			const startTime = performance.now()
			const modelId = getModelId(t.apiConfiguration)

			let bgInputTokens = currentTokens.input
			let bgOutputTokens = currentTokens.output
			let bgCacheWriteTokens = currentTokens.cacheWrite
			let bgCacheReadTokens = currentTokens.cacheRead
			let bgTotalCost = currentTokens.total

			const captureUsageData = async (
				tokens: {
					input: number
					output: number
					cacheWrite: number
					cacheRead: number
					total?: number
				},
				messageIndex: number = apiReqIndex,
			) => {
				if (tokens.input > 0 || tokens.output > 0 || tokens.cacheWrite > 0 || tokens.cacheRead > 0) {
					inputTokens = tokens.input
					outputTokens = tokens.output
					cacheWriteTokens = tokens.cacheWrite
					cacheReadTokens = tokens.cacheRead
					totalCost = tokens.total

					updateApiReqMsg()
					await t.saveClineMessages()

					const apiReqMessage = t.clineMessages[messageIndex]
					if (apiReqMessage) {
						await t.updateClineMessage(apiReqMessage)
					}

					const modelId = getModelId(t.apiConfiguration)
					const apiProvider = t.apiConfiguration.apiProvider
					const apiProtocol = getApiProtocol(
						apiProvider && !isRetiredProvider(apiProvider) ? apiProvider : undefined,
						modelId,
					)

					const _costResult =
						apiProtocol === "anthropic"
							? calculateApiCostAnthropic(
									streamModelInfo,
									tokens.input,
									tokens.output,
									tokens.cacheWrite,
									tokens.cacheRead,
								)
							: calculateApiCostOpenAI(
									streamModelInfo,
									tokens.input,
									tokens.output,
									tokens.cacheWrite,
									tokens.cacheRead,
								)

					globalCacheMetrics.record({
						timestamp: Date.now(),
						inputTokens: tokens.input,
						cacheCreationInputTokens: tokens.cacheWrite,
						cacheReadInputTokens: tokens.cacheRead,
						outputTokens: tokens.output,
						model: cachedModelId,
					})
					t.requestCacheReadWindow.push(tokens.cacheRead)
					if (t.requestCacheReadWindow.length > 5) {
						t.requestCacheReadWindow.shift()
					}
					t.requestInputTokensWindow.push(tokens.input)
					if (t.requestInputTokensWindow.length > 5) {
						t.requestInputTokensWindow.shift()
					}
					const cacheSummary = globalCacheMetrics.getSummary()

					const latencyMs = Date.now() - requestStartedAt
					logger.info(
						"TaskStreamConsumer",
						`Task Metrics: task=${t.taskId} mode=${await t.getTaskMode()} latencyMs=${latencyMs} input=${tokens.input} output=${tokens.output} cacheCreate=${tokens.cacheWrite} cacheRead=${tokens.cacheRead} contextTokens=${t.getTokenUsage().contextTokens ?? 0} cacheHitRate=${cacheSummary.cacheHitRate.toFixed(3)} estSavings=${(cacheSummary.estimatedSavingsPercent * 100).toFixed(1)}% requests=${cacheSummary.totalRequests}`,
					)
					const runtimeTokenUsage = t.getTokenUsage()
					const runtimeContextTokens = runtimeTokenUsage.contextTokens ?? 0
					const runtimeModel = t.api.getModel().info
					const runtimeContextWindow = runtimeModel?.contextWindow ?? 0
					const runtimeContextPercent =
						runtimeContextWindow > 0 ? (runtimeContextTokens / runtimeContextWindow) * 100 : 0
					const runtimeState = await t.hostRef.deref()?.getState()
					const runtimeAutoCondenseContext = runtimeState?.autoCondenseContext ?? true
					const runtimeAutoCondenseContextPercent =
						runtimeState?.autoCondenseContextPercent ?? DEFAULT_AUTO_CONDENSE_CONTEXT_PERCENT
					const runtimeProfileThresholds = runtimeState?.profileThresholds ?? {}
					const runtimeCurrentProfileId = runtimeState?.currentApiConfigName ?? "default"
					const compactLikelyTriggered = willManageContext({
						totalTokens: runtimeContextTokens,
						contextWindow: runtimeContextWindow,
						maxTokens: runtimeModel?.maxTokens,
						autoCondenseContext: runtimeAutoCondenseContext,
						autoCondenseContextPercent: runtimeAutoCondenseContextPercent,
						profileThresholds: runtimeProfileThresholds,
						currentProfileId: runtimeCurrentProfileId,
						lastMessageTokens: 0,
					})
					await t.notifier?.postMessageToWebview({
						type: "taskMetrics",
						taskMetrics: {
							taskId: t.taskId,
							latencyMs,
							cacheHitRate: cacheSummary.cacheHitRate,
							estimatedSavingsPercent: cacheSummary.estimatedSavingsPercent * 100,
							cacheReadInputTokens: tokens.cacheRead,
							cacheCreationInputTokens: tokens.cacheWrite,
							inputTokens: tokens.input,
							outputTokens: tokens.output,
							contextTokens: runtimeContextTokens,
							contextPercent: runtimeContextPercent,
							compactLikelyTriggered,
							cacheBreaksTotal: globalPromptCacheBreakDetector.getTotalBreaks(),
							cacheBreaksBySource: globalPromptCacheBreakDetector.getBreaksBySource(),
						},
					})
				}
			}

			try {
				let usageFound = false
				let chunkCount = 0

				while (!item.done) {
					if (performance.now() - startTime > timeoutMs) {
						logger.warn(
							"TaskStreamConsumer",
							`Background Usage Collection timed out after ${timeoutMs}ms for model: ${modelId}, processed ${chunkCount} chunks`,
						)
						if (iterator.return) {
							await iterator.return(undefined)
						}
						break
					}

					const chunk = item.value
					item = await iterator.next()
					chunkCount++

					if (chunk && chunk.type === "usage") {
						usageFound = true
						bgInputTokens += chunk.inputTokens
						bgOutputTokens += chunk.outputTokens
						bgCacheWriteTokens += chunk.cacheWriteTokens ?? 0
						bgCacheReadTokens += chunk.cacheReadTokens ?? 0
						bgTotalCost = chunk.totalCost
					}
				}

				if (
					usageFound ||
					bgInputTokens > 0 ||
					bgOutputTokens > 0 ||
					bgCacheWriteTokens > 0 ||
					bgCacheReadTokens > 0
				) {
					await captureUsageData(
						{
							input: bgInputTokens,
							output: bgOutputTokens,
							cacheWrite: bgCacheWriteTokens,
							cacheRead: bgCacheReadTokens,
							total: bgTotalCost,
						},
						lastApiReqIndex,
					)
				} else {
					logger.warn(
						"TaskStreamConsumer",
						`Background Usage Collection: request ${apiReqIndex} is complete, but no usage info was found. Model: ${modelId}`,
					)
				}
			} catch (error) {
				logger.error("TaskStreamConsumer", "Error draining stream for usage data:", error)
				TelemetryService.reportError(
					error instanceof Error ? error : new Error(String(error)),
					TelemetryEventName.UTILITY_ERROR,
				)
				if (bgInputTokens > 0 || bgOutputTokens > 0 || bgCacheWriteTokens > 0 || bgCacheReadTokens > 0) {
					await captureUsageData(
						{
							input: bgInputTokens,
							output: bgOutputTokens,
							cacheWrite: bgCacheWriteTokens,
							cacheRead: bgCacheReadTokens,
							total: bgTotalCost,
						},
						lastApiReqIndex,
					)
				}
			}
		}

		drainStreamInBackgroundToFindAllUsage(lastApiReqIndex).catch((error) => {
			logger.error("TaskStreamConsumer", "Background usage collection failed:", error)
			TelemetryService.reportError(
				error instanceof Error ? error : new Error(String(error)),
				TelemetryEventName.UTILITY_ERROR,
			)
		})
	} catch (error) {
		if (!t.abandoned) {
			const rawErrorMessage = getErrorMessage(error)
			const streamingFailedMessage = t.abort
				? undefined
				: `${i18nT("common:interruption.streamTerminatedByProvider")}: ${rawErrorMessage}`

			const retryAction = await handleMidStreamFailure({
				task: t,
				error,
				currentRetryAttempt: retryAttempt,
				currentUserContent,
				stack,
				streamingFailedMessage,
				abortStream,
			})

			if (retryAction === "continue") {
				streamAction = "continue"
			} else if (retryAction === "break") {
				streamAction = "break"
			}
		}
	} finally {
		t.isStreaming = false
		const profile = globalQueryProfiler.finish(requestProfileId, {
			aborted: t.abort || t.abandoned,
		})
		if (profile) {
			logger.info(
				"TaskStreamConsumer",
				`Query Profiler: task=${profile.taskId} model=${profile.modelId} ttftMs=${profile.ttftMs ?? -1} e2eMs=${profile.e2eMs ?? -1} aborted=${profile.aborted}`,
			)
			// Report query performance to telemetry (Task 2.2)
			try {
				TelemetryService.instance.captureEvent("query.completed", {
					model: profile.modelId,
					ttft: profile.ttftMs ?? -1,
					e2e: profile.e2eMs ?? -1,
					success: !profile.aborted && !profile.error,
				})
			} catch {
				// Telemetry failure is non-fatal
			}
		}
		t.currentRequestAbortController = undefined
	}

	return { assistantMessage, reasoningMessage, pendingGroundingSources, action: streamAction }
}

export interface FinalizeConfig {
	task: TaskExecutorHost
	toolCallParser: NativeToolCallParser
	placeFinalizedStreamingToolUse: FinalizeToolUseFn
	consumptionResult: StreamConsumptionResult
	requestProfileId: string
	lastApiReqIndex: number
	retryAttempt: number
	currentUserContent: Anthropic.Messages.ContentBlockParam[]
	stack: StackItem[]
}

export interface FinalizeResult {
	action: "continue" | "break" | "done"
}

export async function finalizeStreamResponse(config: FinalizeConfig): Promise<FinalizeResult> {
	const {
		task: t,
		toolCallParser,
		placeFinalizedStreamingToolUse,
		consumptionResult,
		requestProfileId,
		lastApiReqIndex: _lastApiReqIndex,
		retryAttempt,
		currentUserContent,
		stack,
	} = config

	const { assistantMessage, reasoningMessage, pendingGroundingSources } = consumptionResult

	if (t.abort || t.abandoned) {
		t.stateMachine.force(TaskState.ERROR)
		throw new TaskAbortedError(t.taskId, t.instanceId)
	}

	t.didCompleteReadingStream = true

	await finalizePendingStreamingToolCalls({
		task: t,
		toolCallParser,
		finalizeToolUse: placeFinalizedStreamingToolUse,
	})

	const partialBlocks = t.assistantMessageContent.filter((block) => block.partial)
	partialBlocks.forEach((block) => (block.partial = false))

	if (reasoningMessage) {
		const lastReasoningIndex = findLastIndex(
			t.clineMessages,
			(m: ClineMessage) => m.type === "say" && m.say === "reasoning",
		)

		if (lastReasoningIndex !== -1 && t.clineMessages[lastReasoningIndex]!.partial) {
			t.clineMessages[lastReasoningIndex]!.partial = false
			await t.updateClineMessage(t.clineMessages[lastReasoningIndex]!)
		}
	}

	if (!t._savedMessagesForCurrentRequest) {
		await t.saveClineMessages()
		await t.refreshWebviewState()
	}

	const hasTextContent = assistantMessage.length > 0

	const hasToolUses = t.assistantMessageContent.some(isAnyToolUse)

	if (hasTextContent || hasToolUses) {
		t.consecutiveNoAssistantMessagesCount = 0
		if (pendingGroundingSources.length > 0) {
			const citationLinks = pendingGroundingSources.map(
				(source: GroundingSource, i: number) => `[${i + 1}](${source.url})`,
			)
			const sourcesText = `${i18nT("common:gemini.sources")} ${citationLinks.join(", ")}`

			await t.say("text", sourcesText, undefined, false, undefined, undefined, {
				isNonInteractive: true,
			})
		}

		const assistantContent: Array<Anthropic.TextBlockParam | Anthropic.ToolUseBlockParam> = []

		if (assistantMessage) {
			assistantContent.push({
				type: "text" as const,
				text: assistantMessage,
			})
		}

		const seenToolUseIds = new Set<string>()
		const toolUseBlocks = t.assistantMessageContent.filter(isAnyToolUse)
		for (const block of toolUseBlocks) {
			if (block.type === "mcp_tool_use") {
				const mcpBlock = block as import("../../shared/tools").McpToolUse
				if (mcpBlock.id) {
					const sanitizedId = sanitizeToolUseId(mcpBlock.id)
					if (seenToolUseIds.has(sanitizedId)) {
						logger.warn(
							"TaskStreamConsumer",
							`Pre-flight deduplication: Skipping duplicate MCP tool_use ID: ${sanitizedId} (tool: ${mcpBlock.name}) on task ${t.taskId}`,
						)
						continue
					}
					seenToolUseIds.add(sanitizedId)
					assistantContent.push({
						type: "tool_use" as const,
						id: sanitizedId,
						name: mcpBlock.name,
						input: mcpBlock.arguments,
					})
				}
			} else {
				const toolUse = block as import("../../shared/tools").ToolUse
				const toolCallId = toolUse.id
				if (toolCallId) {
					const sanitizedId = sanitizeToolUseId(toolCallId)
					if (seenToolUseIds.has(sanitizedId)) {
						logger.warn(
							"TaskStreamConsumer",
							`Pre-flight deduplication: Skipping duplicate tool_use ID: ${sanitizedId} (tool: ${toolUse.name}) on task ${t.taskId}`,
						)
						continue
					}
					seenToolUseIds.add(sanitizedId)
					const input = toolUse.nativeArgs || toolUse.params

					const toolNameForHistory = toolUse.originalName ?? toolUse.name

					assistantContent.push({
						type: "tool_use" as const,
						id: sanitizedId,
						name: toolNameForHistory,
						input,
					})
				}
			}
		}

		const newTaskIndex = assistantContent.findIndex(
			(block) => block.type === "tool_use" && block.name === "new_task",
		)

		if (newTaskIndex !== -1 && newTaskIndex < assistantContent.length - 1) {
			const truncatedTools = assistantContent.slice(newTaskIndex + 1)
			assistantContent.length = newTaskIndex + 1

			const executionNewTaskIndex = t.assistantMessageContent.findIndex(
				(block) => isToolUseBlock(block) && block.name === "new_task",
			)
			if (executionNewTaskIndex !== -1) {
				t.assistantMessageContent.length = executionNewTaskIndex + 1
			}

			for (const tool of truncatedTools) {
				if (tool.type === "tool_use" && (tool as Anthropic.ToolUseBlockParam).id) {
					t.pushToolResultToUserContent({
						type: "tool_result",
						tool_use_id: (tool as Anthropic.ToolUseBlockParam).id,
						content:
							"This tool was not executed because new_task was called in the same message turn. The new_task tool must be the last tool in a message.",
						is_error: true,
					})
				}
			}
		}

		await t.addToApiConversationHistory(
			{ role: "assistant", content: assistantContent },
			reasoningMessage || undefined,
		)
		t.assistantMessageSavedToHistory = true

		// MemRL: record assistant step to STM (best-effort, never blocks)
		try {
			const provider = t.hostRef?.deref()
			const mm = provider?.getMemoryManager?.(t.cwd)
			if (mm) {
				const stm = mm.getStm(t.taskId)
				const textSummary = assistantContent
					.filter((b): b is { type: "text"; text: string } => b.type === "text")
					.map((b) => b.text)
					.join(" ")
					.trim()
					.slice(0, 200)
				if (textSummary) stm.push("assistant", textSummary)
			}
		} catch {
			// silent — STM population must never affect task flow
		}
	}

	if (partialBlocks.length > 0) {
		void t.presentAssistantMessage().catch((error) => {
			logger.error("presentAssistantMessage failed", error)
			TelemetryService.reportError(
				error instanceof Error ? error : new Error(String(error)),
				TelemetryEventName.UTILITY_ERROR,
			)
		})
	}

	if (hasTextContent || hasToolUses) {
		const waitStartMs = performance.now()
		debugLog(
			`[TaskStreamConsumer][${requestProfileId}] pWaitFor userMessageContentReady – start (contentIndex=${t.currentStreamingContentIndex}, blocks=${t.assistantMessageContent.length})`,
		)

		try {
			await pWaitFor(() => t.userMessageContentReady, {
				...(USER_MESSAGE_CONTENT_READY_TIMEOUT_MS > 0 && {
					timeout: USER_MESSAGE_CONTENT_READY_TIMEOUT_MS,
				}),
			})
		} catch (_timeoutErr) {
			const pendingToolBlocks = t.assistantMessageContent.filter(
				(b: UnsafeAny) =>
					(b.type === "tool_use" || b.type === "mcp_tool_use") &&
					b.id &&
					!t.userMessageContent.some(
						(r: UnsafeAny) => r.type === "tool_result" && r.tool_use_id === sanitizeToolUseId(b.id),
					),
			)

			const isAttemptCompletionWaitingForUser =
				pendingToolBlocks.some((b: UnsafeAny) => b.name === "attempt_completion") &&
				t.clineMessages.some(
					(m: UnsafeAny) => m.type === "ask" && m.ask === "completion_result" && !m.isAnswered,
				)

			if (isAttemptCompletionWaitingForUser) {
				logger.info(
					"TaskStreamConsumer",
					"userMessageContentReady timed out, but attempt_completion is waiting for user confirmation. Continuing to wait without timeout.",
				)

				await pWaitFor(() => t.userMessageContentReady || t.abort || t.abandoned || t.taskCompleted)

				if (t.abort || t.abandoned) {
					t.stateMachine.force(TaskState.ERROR)
					throw new TaskAbortedError(t.taskId, t.instanceId)
				}
			} else {
				logger.error(
					"TaskStreamConsumer",
					`userMessageContentReady timed out after ${USER_MESSAGE_CONTENT_READY_TIMEOUT_MS}ms ` +
						`(taskId=${t.taskId}, instance=${t.instanceId}, contentIndex=${t.currentStreamingContentIndex}, ` +
						`blocks=${t.assistantMessageContent.length}, pendingTools=${pendingToolBlocks.length})`,
				)

				for (const block of pendingToolBlocks) {
					const toolUseId = (block as ToolUse | McpToolUse).id ?? ""
					t.pushToolResultToUserContent({
						type: "tool_result",
						tool_use_id: sanitizeToolUseId(toolUseId),
						content: formatResponse.toolError(
							"Tool execution timed out — the handler did not return a result within the allowed window.",
						),
						is_error: true,
					})
				}

				markUserContentReadyIfDrained(t as UnsafeAny)
				t.userMessageContentReady = true
			}
		}

		debugLog(
			`[TaskStreamConsumer][${requestProfileId}] pWaitFor userMessageContentReady – done (${(performance.now() - waitStartMs).toFixed(0)}ms)`,
		)

		const didToolUse = t.assistantMessageContent.some(isAnyToolUse)

		if (!didToolUse) {
			if (t.taskCompleted) {
				return { action: "done" }
			}

			t.consecutiveNoToolUseCount++

			if (t.consecutiveNoToolUseCount >= 3) {
				await t.say("error", "MODEL_NO_TOOLS_USED")
				t.consecutiveMistakeCount += 2
				t.userMessageContent.push({
					type: "text",
					text: formatResponse.toolRetryThrottled(),
				})
				t.consecutiveNoToolUseCount = 0
			} else if (t.consecutiveNoToolUseCount >= 2) {
				await t.say("error", "MODEL_NO_TOOLS_USED")
				t.consecutiveMistakeCount++

				const lastUserMsg = t.apiConversationHistory.filter((m: UnsafeAny) => m.role === "user").pop()
				const wasInterrupted =
					lastUserMsg &&
					Array.isArray(lastUserMsg.content) &&
					lastUserMsg.content.some(
						(block) =>
							(block as TypedBlock).type === "tool_result" &&
							typeof (block as TypedBlock).content === "string" &&
							((block as TypedBlock).content as string).includes("interrupted"),
					)

				t.userMessageContent.push({
					type: "text",
					text: wasInterrupted ? formatResponse.noToolsUsedWithInterruptHint() : formatResponse.noToolsUsed(),
				})
			} else {
				t.userMessageContent.push({
					type: "text",
					text: formatResponse.noToolsUsed(),
				})
			}
		} else {
			t.consecutiveNoToolUseCount = 0
		}

		if (t.taskCompleted) {
			return { action: "done" }
		}

		if (t.userMessageContent.length > 0 || t.isPaused) {
			stack.push({
				userContent: [...t.userMessageContent],
				includeFileDetails: false,
			})

			await new Promise((resolve) => setImmediate(resolve))
		}

		return { action: "continue" }
	} else {
		const emptyResponseAction = await handleEmptyAssistantResponse({
			task: t,
			currentRetryAttempt: retryAttempt,
			currentUserContent,
			stack,
		})

		if (emptyResponseAction === "continue") {
			return { action: "continue" }
		}
		if (emptyResponseAction === "break") {
			return { action: "break" }
		}
	}

	return { action: "done" }
}
