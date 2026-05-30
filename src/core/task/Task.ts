import * as path from "path"
import os from "os"
import crypto from "crypto"
import { v7 as uuidv7 } from "uuid"
import EventEmitter from "events"

import { startAllPrefetch } from "../prefetch"
import { setLastGlobalApiRequestTime, getLastGlobalApiRequestTime as getGlobalApiTime } from "./globalApiTiming"

import { Anthropic } from "@anthropic-ai/sdk"
import debounce from "lodash.debounce"

import {
	type TaskLike,
	type TaskMetadata,
	type TaskEvents,
	type ProviderSettings,
	type TokenUsage,
	type ToolUsage,
	type ToolName,
	type ContextCondense,
	type ContextTruncation,
	type ClineMessage,
	type ClineSay,
	type ClineAsk,
	type ToolProgressStatus,
	type HistoryItem,
	type CreateTaskOptions,
	type ModelInfo,
	type ClineApiReqCancelReason,
	NJUST_AI_CJEventName,
	TelemetryEventName,
	TaskStatus,
	TodoItem,
	QueuedMessage,
	DEFAULT_CONSECUTIVE_MISTAKE_LIMIT,
	DEFAULT_CHECKPOINT_TIMEOUT_SECONDS,
	MAX_CHECKPOINT_TIMEOUT_SECONDS,
	MIN_CHECKPOINT_TIMEOUT_SECONDS,
	countEnabledMcpTools,
} from "@njust-ai-cj/types"
import { TelemetryService } from "@njust-ai-cj/telemetry"

// api
import { ApiHandler, buildApiHandler } from "../../api"
import { ApiStream } from "../../api/transform/stream"

// shared
import { findLastIndex } from "../../shared/array"
import { combineApiRequests } from "../../shared/combineApiRequests"
import { combineCommandSequences } from "../../shared/combineCommandSequences"
import { getApiMetrics, hasTokenUsageChanged, hasToolUsageChanged } from "../../shared/getApiMetrics"
import { ClineAskResponse } from "../../shared/WebviewMessage"
import { defaultModeSlug } from "../../shared/modes"
import { DiffStrategy } from "../../shared/tools"
import { logger } from "../../shared/logger"

import { RepoPerTaskCheckpointService } from "../../services/checkpoints"

import { RooTerminalProcess } from "../../integrations/terminal/types"

// utils
import { getWorkspacePath } from "../../utils/path"
import type { ICloudAgentHost } from "./interfaces/ICloudAgentHost"
import { tokenCountCache } from "../../utils/tokenCountCache"

// prompts
import { formatResponse } from "../prompts/responses"
import { type SystemPromptParts } from "../prompts/system"

// core modules
import { ToolRepetitionDetector } from "../tools/ToolRepetitionDetector"

import { FileContextTracker } from "../context-tracking/FileContextTracker"
import { RooIgnoreController } from "../ignore/RooIgnoreController"
import { RooProtectedController } from "../protect/RooProtectedController"
import { type AssistantMessageContent } from "../assistant-message/types"
import { NativeToolCallParser } from "../assistant-message/NativeToolCallParser"
import { ToolExecutionContext } from "./ToolExecutionContext"
import { TokenGrowthTracker } from "../context-management/tokenGrowthTracker"
import { taskEventBus, type TaskEventBus } from "../events/TaskEventBus"
import type { ITaskHost } from "./interfaces/ITaskHost"
import type { ITaskUINotifier } from "./interfaces/ITaskUINotifier"
import { NullTaskDiffViewProvider, type ITaskDiffViewProvider } from "./interfaces/ITaskDiffViewProvider"
import { MultiSearchReplaceDiffStrategy } from "../diff/strategies/multi-search-replace"
import { type ApiMessage } from "../task-persistence"
import { ErrorRecoveryHandler } from "./ErrorRecoveryHandler"
import { PersistentRetryManager } from "./PersistentRetry"
import {
	type CheckpointDiffOptions,
	type CheckpointRestoreOptions,
	getCheckpointService,
	checkpointSave,
	checkpointRestore,
	checkpointDiff,
} from "../checkpoints"
import { MessageQueueService } from "../message-queue/MessageQueueService"
import { AutoApprovalHandler } from "../auto-approval"
import { MessageManager } from "../message-manager"
import { TaskStateMachine, TaskState } from "./TaskStateMachine"
import { TaskRequestBuilder } from "./TaskRequestBuilder"
import { TaskStreamProcessor } from "./TaskStreamProcessor"
import type { IsolationLevel, ForkedContextConfig, CacheSafeParams } from "./SubTaskOptions"
import { TaskMessageManager, type TaskMessageContext } from "./TaskMessageManager"
import { TaskAskSayHandler } from "./TaskAskSayHandler"
import { type TaskAskSayHost } from "./interfaces/TaskAskSayHost"
import { TaskSubtaskHandler } from "./TaskSubtaskHandler"
import { type TaskSubtaskHost } from "./interfaces/TaskSubtaskHost"
import { TaskToolHandler } from "./TaskToolHandler"
import { TaskExecutor, type TaskExecutorHost } from "./TaskExecutor"
import { TaskLifecycleHandler, type TaskLifecycleHost } from "./TaskLifecycleHandler"
import { CangjieRuntimePolicy } from "./CangjieRuntimePolicy"
import { getErrorMessage } from "../../shared/error-utils"
import {
	addToApiConversationHistoryWithTask,
	addToClineMessagesWithTask,
	findMessageByIdWithTask,
	findMessageByTimestampWithTask,
	flushPendingToolResultsToHistoryWithTask,
	getSavedApiConversationHistoryWithTask,
	getSavedClineMessagesWithTask,
	overwriteApiConversationHistoryWithTask,
	overwriteClineMessagesWithTask,
	retrySaveApiConversationHistoryWithTask,
	saveApiConversationHistoryWithTask,
	saveClineMessagesWithTask,
	updateClineMessageWithTask,
} from "./TaskPersistence"

export interface TaskOptions extends CreateTaskOptions {
	host?: ITaskHost
	/** @deprecated Use {@link host}. Accepted for backward compatibility with tests/call sites. */
	provider?: ITaskHost
	/** Optional override; defaults to shared {@link taskEventBus}. */
	eventBus?: TaskEventBus
	apiConfiguration: ProviderSettings
	enableCheckpoints?: boolean
	checkpointTimeout?: number
	consecutiveMistakeLimit?: number
	task?: string
	images?: string[]
	historyItem?: HistoryItem
	experiments?: Record<string, boolean>
	startTask?: boolean
	rootTask?: Task
	parentTask?: Task
	taskNumber?: number
	onCreated?: (task: Task) => void
	initialTodos?: TodoItem[]
	workspacePath?: string
	/** Capability-scoped tool whitelist for this task (used for delegated child tasks). */
	allowedTools?: string[]
	/** Optional trace id used to stitch parent/child task observability spans. */
	parentTraceId?: string
	/** Initial status for the task's history item (e.g., "active" for child tasks) */
	initialStatus?: "active" | "delegated" | "completed"
}

export class Task extends EventEmitter<TaskEvents> implements TaskLike {
	readonly taskId: string
	readonly rootTaskId?: string
	readonly parentTaskId?: string
	childTaskId?: string
	pendingNewTaskToolCallId?: string

	readonly instanceId: string
	readonly metadata: TaskMetadata

	todoList?: TodoItem[]

	readonly rootTask: Task | undefined = undefined
	readonly parentTask: Task | undefined = undefined
	readonly taskNumber: number
	readonly workspacePath: string
	readonly allowedTools?: ReadonlySet<string>
	readonly parentTraceId?: string

	/** Forked context summary injected by parent when isolationLevel is "forked" */
	forkedContextSummary?: string
	/** Isolation level for this task (set when created as a sub-task) */
	isolationLevel?: IsolationLevel

	/**
	 * Cached tool definitions to avoid rebuilding for the same mode.
	 * Shared from parent to child via inheritCacheFromParent.
	 */
	public cachedToolDefinitions?: { mode: string; tools: UnsafeAny[]; time: number }

	/** MemRL: episodic hints retrieved before this task run, injected into system prompt. */
	public memrlEpisodicHints: string = ""
	/** MemRL: learned LTM rule cards retrieved before this task run, injected into system prompt. */
	public memrlLtmRules: string = ""

	/**
	 * The mode associated with this task. Persisted across sessions
	 * to maintain user context when reopening tasks from history.
	 *
	 * ## Lifecycle
	 *
	 * ### For new tasks:
	 * 1. Initially `undefined` during construction
	 * 2. Asynchronously initialized from provider state via `initializeTaskMode()`
	 * 3. Falls back to `defaultModeSlug` if provider state is unavailable
	 *
	 * ### For history items:
	 * 1. Immediately set from `historyItem.mode` during construction
	 * 2. Falls back to `defaultModeSlug` if mode is not stored in history
	 *
	 * ## Important
	 * This property should NOT be accessed directly until `taskModeReady` promise resolves.
	 * Use `getTaskMode()` for async access or `taskMode` getter for sync access after initialization.
	 *
	 * @private
	 * @see {@link getTaskMode} - For safe async access
	 * @see {@link taskMode} - For sync access after initialization
	 * @see {@link waitForModeInitialization} - To ensure initialization is complete
	 */
	private _taskMode: string | undefined

	/**
	 * Promise that resolves when the task mode has been initialized.
	 * This ensures async mode initialization completes before the task is used.
	 *
	 * ## Purpose
	 * - Prevents race conditions when accessing task mode
	 * - Ensures provider state is properly loaded before mode-dependent operations
	 * - Provides a synchronization point for async initialization
	 *
	 * ## Resolution timing
	 * - For history items: Resolves immediately (sync initialization)
	 * - For new tasks: Resolves after provider state is fetched (async initialization)
	 *
	 * @private
	 * @see {@link waitForModeInitialization} - Public method to await this promise
	 */
	private taskModeReady: Promise<void>

	/**
	 * The API configuration name (provider profile) associated with this task.
	 * Persisted across sessions to maintain the provider profile when reopening tasks from history.
	 *
	 * ## Lifecycle
	 *
	 * ### For new tasks:
	 * 1. Initially `undefined` during construction
	 * 2. Asynchronously initialized from provider state via `initializeTaskApiConfigName()`
	 * 3. Falls back to "default" if provider state is unavailable
	 *
	 * ### For history items:
	 * 1. Immediately set from `historyItem.apiConfigName` during construction
	 * 2. Falls back to undefined if not stored in history (for backward compatibility)
	 *
	 * ## Important
	 * If you need a non-`undefined` provider profile (e.g., for profile-dependent operations),
	 * wait for `taskApiConfigReady` first (or use `getTaskApiConfigName()`).
	 * The sync `taskApiConfigName` getter may return `undefined` for backward compatibility.
	 *
	 * @private
	 * @see {@link getTaskApiConfigName} - For safe async access
	 * @see {@link taskApiConfigName} - For sync access after initialization
	 */
	private _taskApiConfigName: string | undefined

	/**
	 * Promise that resolves when the task API config name has been initialized.
	 * This ensures async API config name initialization completes before the task is used.
	 *
	 * ## Purpose
	 * - Prevents race conditions when accessing task API config name
	 * - Ensures provider state is properly loaded before profile-dependent operations
	 * - Provides a synchronization point for async initialization
	 *
	 * ## Resolution timing
	 * - For history items: Resolves immediately (sync initialization)
	 * - For new tasks: Resolves after provider state is fetched (async initialization)
	 *
	 * @private
	 */
	private taskApiConfigReady: Promise<void>

	hostRef: WeakRef<ITaskHost>
	private readonly eventBus: TaskEventBus

	/** @deprecated Use hostRef (same WeakRef). Presents Task host for legacy call sites. */
	get providerRef(): WeakRef<ITaskHost> {
		return this.hostRef
	}

	/** Narrow accessor: returns only the UI notification facet of the host. */
	private get notifier(): ITaskUINotifier | undefined {
		return this.hostRef.deref()
	}

	/** Emit a UI state refresh. Fires event bus + calls host notifier. */
	protected async refreshWebviewState(): Promise<void> {
		this.eventBus.emit("task:tokens-updated", { taskId: this.taskId })
		await this.notifier?.postStateToWebviewWithoutTaskHistory()
	}

	readonly globalStoragePath: string
	abort: boolean = false

	// ── Foreground→Background switching ──
	/** Resolved when the user requests this task be moved to background */
	private _backgroundResolve: ((value: void) => void) | null = null
	/** Promise that resolves when background switch is requested */
	private _backgroundSignal: Promise<void> | null = null
	/** Whether this task has been switched to background */
	isBackgrounded: boolean = false
	/** Set true when attempt_completion is accepted. The outer loop in
	 *  initiateTaskLoop checks this flag to stop re-prompting the model
	 *  after a completed task. */
	taskCompleted: boolean = false
	private persistentRetryHandler?: PersistentRetryManager
	currentRequestAbortController?: AbortController
	skipPrevResponseIdOnce: boolean = false

	// TaskStatus
	idleAsk?: ClineMessage
	resumableAsk?: ClineMessage
	interactiveAsk?: ClineMessage

	// Context management circuit breaker
	compactFailureCount = 0
	readonly maxCompactFailures = 3

	didFinishAbortingStream = false
	abandoned = false
	abortReason?: ClineApiReqCancelReason
	isInitialized = false
	isPaused: boolean = false

	// API
	apiConfiguration: ProviderSettings
	api: ApiHandler
	// Note: lastGlobalApiRequestTime is now managed in globalApiTiming.ts
	private autoApprovalHandler: AutoApprovalHandler

	/**
	 * Reset the global API request timestamp. This should only be used for testing.
	 * @internal
	 */
	static resetGlobalApiRequestTime(): void {
		setLastGlobalApiRequestTime(undefined)
	}

	/**
	 * Get the last global API request timestamp.
	 * Exposed for TaskStreamProcessor (delegation pattern).
	 */
	static getLastGlobalApiRequestTime(): number | undefined {
		return getGlobalApiTime()
	}

	toolRepetitionDetector: ToolRepetitionDetector
	rooIgnoreController?: RooIgnoreController
	rooProtectedController?: RooProtectedController
	// compactFailureCount lives at line 319 (ErrorRecoveryHandler uses it)
	toolCallParser = new NativeToolCallParser()
	fileContextTracker: FileContextTracker
	terminalProcess?: RooTerminalProcess

	// Editing
	diffViewProvider: ITaskDiffViewProvider
	diffStrategy?: DiffStrategy
	didEditFile: boolean = false

	/** Std library module names observed via `search_files` on bundled Cangjie corpus (cangjie mode). Used for write/edit search-gate warnings. */
	cangjieSearchHistory: Set<string> = new Set()
	readonly cangjieRuntimePolicy: CangjieRuntimePolicy

	// LLM Messages & Chat Messages
	apiConversationHistory: ApiMessage[] = []
	clineMessages: ClineMessage[] = []

	// Ask
	private askResponse?: ClineAskResponse
	private askResponseText?: string
	private askResponseImages?: string[]
	public lastMessageTs?: number
	private autoApprovalTimeoutRef?: NodeJS.Timeout

	// Tool Use
	consecutiveMistakeCount: number = 0
	consecutiveMistakeLimit: number
	consecutiveMistakeCountForApplyDiff: Map<string, number> = new Map()
	consecutiveMistakeCountForEditFile: Map<string, number> = new Map()
	consecutiveNoToolUseCount: number = 0
	consecutiveNoAssistantMessagesCount: number = 0
	toolUsage: ToolUsage = {}

	// Checkpoints
	enableCheckpoints: boolean
	checkpointTimeout: number
	checkpointService?: RepoPerTaskCheckpointService
	checkpointServiceInitializing = false

	// Message Queue Service
	public readonly messageQueueService: MessageQueueService
	private messageQueueStateChangedHandler: (() => void) | undefined

	// Streaming
	isWaitingForFirstChunk = false
	isStreaming = false
	currentStreamingContentIndex = 0
	currentStreamingDidCheckpoint = false
	assistantMessageContent: AssistantMessageContent[] = []
	presentAssistantMessageLocked = false
	presentAssistantMessageHasPendingUpdates = false
	userMessageContent: (Anthropic.TextBlockParam | Anthropic.ImageBlockParam | Anthropic.ToolResultBlockParam)[] = []
	userMessageContentReady = false

	/**
	 * Flag indicating whether the assistant message for the current streaming session
	 * has been saved to API conversation history.
	 *
	 * This is critical for parallel tool calling: tools should NOT execute until
	 * the assistant message is saved. Otherwise, if a tool like `new_task` triggers
	 * `flushPendingToolResultsToHistory()`, the user message with tool_results would
	 * appear BEFORE the assistant message with tool_uses, causing API errors.
	 *
	 * Reset to `false` at the start of each API request.
	 * Set to `true` after the assistant message is saved in `recursivelyMakeClineRequests`.
	 */
	assistantMessageSavedToHistory = false

	/**
	 * Push a tool_result block to userMessageContent, preventing duplicates.
	 * Duplicate tool_use_ids cause API errors.
	 *
	 * @param toolResult - The tool_result block to add
	 * @returns true if added, false if duplicate was skipped
	 */
	public pushToolResultToUserContent(toolResult: Anthropic.ToolResultBlockParam): boolean {
		const existingResult = this.userMessageContent.find(
			(block): block is Anthropic.ToolResultBlockParam =>
				block.type === "tool_result" && block.tool_use_id === toolResult.tool_use_id,
		)
		if (existingResult) {
			logger.warn(
				"Task",
				`pushToolResultToUserContent: Skipping duplicate tool_result for tool_use_id: ${toolResult.tool_use_id}`,
			)
			return false
		}
		this.userMessageContent.push(toolResult)
		return true
	}
	didRejectTool = false
	didAlreadyUseTool = false
	didToolFailInCurrentTurn = false
	didCompleteReadingStream = false
	private _started = false
	// No streaming parser is required.
	assistantMessageParser?: undefined
	private providerProfileChangeListener?: (config: { name: string; provider?: string }) => void

	// Native tool call streaming state (track which index each tool is at)
	private streamingToolCallIndices: Map<string, number> = new Map()
	readonly toolExecution = new ToolExecutionContext(
		Math.max(1, Number(process.env.ROO_MAX_TOOL_CONCURRENCY ?? 10) || 10),
	)
	private requestCacheReadWindow: number[] = []
	private requestInputTokensWindow: number[] = []
	private readonly tokenGrowthTracker = new TokenGrowthTracker({ maxWindowSize: 6, emaAlpha: 0.4 })

	// Cached model info for current streaming session (set at start of each API request)
	// This prevents excessive getModel() calls during tool execution
	cachedStreamingModel?: { id: string; info: ModelInfo }

	// Token Usage Cache
	tokenUsageSnapshot?: TokenUsage
	tokenUsageSnapshotAt?: number

	// Tool Usage Cache
	private toolUsageSnapshot?: ToolUsage

	// Token Usage Throttling - Debounced emit function
	private readonly TOKEN_USAGE_EMIT_INTERVAL_MS = 2000 // 2 seconds
	private debouncedEmitTokenUsage: ReturnType<typeof debounce>

	private queuedMessageTimer?: ReturnType<typeof setTimeout>
	// Delegate for system prompt generation, caching, and context condensation
	readonly requestBuilder: TaskRequestBuilder

	// Delegate for stream-related helpers (rate limiting, backoff, context window recovery, etc.)
	readonly streamProcessor: TaskStreamProcessor

	// Delegate for error classification, recovery strategies, and circuit breaker logic
	readonly errorRecovery: ErrorRecoveryHandler

	// Initial status for the task's history item (set at creation time to avoid race conditions)
	private readonly initialStatus?: "active" | "delegated" | "completed"

	// MessageManager for high-level message operations (lazy initialized)
	private _messageManager?: MessageManager
	private readonly stateMachine = new TaskStateMachine()

	// Extracted sub-modules (Task 7 decomposition)
	private _taskMessageManager?: TaskMessageManager
	private _taskToolHandler?: TaskToolHandler
	private _executor?: TaskExecutor
	private _lifecycleHandler?: TaskLifecycleHandler

	/** @internal Persistence/CRUD operations on messages — delegates to TaskMessageManager. */
	private get msgMgr(): TaskMessageManager {
		if (!this._taskMessageManager) {
			this._taskMessageManager = new TaskMessageManager(this as UnsafeAny as TaskMessageContext)
		}
		return this._taskMessageManager
	}

	/** @internal Ask/say operations — delegates to TaskAskSayHandler. */
	private get askSayHandler(): TaskAskSayHandler {
		if (!this._askSayHandler) {
			this._askSayHandler = new TaskAskSayHandler(this as UnsafeAny as TaskAskSayHost)
		}
		return this._askSayHandler
	}
	private _askSayHandler?: TaskAskSayHandler

	/** @internal Tool result accumulation — delegates to TaskToolHandler. */
	private get toolHandler(): TaskToolHandler {
		if (!this._taskToolHandler) {
			this._taskToolHandler = new TaskToolHandler(this)
		}
		return this._taskToolHandler
	}

	/** @internal API request loop — delegates to TaskExecutor. */
	private get executor(): TaskExecutor {
		if (!this._executor) {
			this._executor = new TaskExecutor(this as UnsafeAny as TaskExecutorHost)
		}
		return this._executor
	}

	/** @internal Lifecycle operations — delegates to TaskLifecycleHandler. */
	private get lifecycleHandler(): TaskLifecycleHandler {
		if (!this._lifecycleHandler) {
			this._lifecycleHandler = new TaskLifecycleHandler(this as UnsafeAny as TaskLifecycleHost)
		}
		return this._lifecycleHandler
	}

	/** @internal Subtask operations — delegates to TaskSubtaskHandler. */
	private get subtaskHandler(): TaskSubtaskHandler {
		if (!this._subtaskHandler) {
			this._subtaskHandler = new TaskSubtaskHandler(this as UnsafeAny as TaskSubtaskHost)
		}
		return this._subtaskHandler
	}
	private _subtaskHandler?: TaskSubtaskHandler

	/** Accessor for the static lastGlobalApiRequestTime (used by TaskExecutor). */
	public setLastGlobalApiRequestTime(time: number): void {
		setLastGlobalApiRequestTime(time)
	}

	/** Accessor for the static lastGlobalApiRequestTime (used by TaskExecutor). */
	public getLastGlobalApiRequestTimeValue(): number | undefined {
		return getGlobalApiTime()
	}

	/** Request assistant message presentation via outer event subscriber. */
	public async presentAssistantMessage(): Promise<void> {
		await this.eventBus.emitAsync("task:assistant-message-requested", {
			taskId: this.taskId,
			data: { task: this },
		})
	}

	public get taskState(): TaskState {
		return this.stateMachine.state
	}

	public forceTaskState(state: TaskState): void {
		this.stateMachine.force(state)
	}

	constructor({
		host: hostOption,
		provider,
		eventBus: injectedEventBus,
		apiConfiguration,
		enableCheckpoints = true,
		checkpointTimeout = DEFAULT_CHECKPOINT_TIMEOUT_SECONDS,
		consecutiveMistakeLimit = DEFAULT_CONSECUTIVE_MISTAKE_LIMIT,
		taskId,
		task,
		images,
		historyItem,
		startTask = true,
		rootTask,
		parentTask,
		taskNumber = -1,
		onCreated,
		initialTodos,
		workspacePath,
		initialStatus,
	}: TaskOptions) {
		super()

		const host = hostOption ?? provider
		if (!host) {
			throw new Error("Task requires host (or legacy provider) option")
		}

		this.eventBus = injectedEventBus ?? taskEventBus

		if (startTask && !task && !images && !historyItem) {
			throw new Error("Either historyItem or task/images must be provided")
		}

		if (
			!checkpointTimeout ||
			checkpointTimeout > MAX_CHECKPOINT_TIMEOUT_SECONDS ||
			checkpointTimeout < MIN_CHECKPOINT_TIMEOUT_SECONDS
		) {
			throw new Error(
				"checkpointTimeout must be between " +
					MIN_CHECKPOINT_TIMEOUT_SECONDS +
					" and " +
					MAX_CHECKPOINT_TIMEOUT_SECONDS +
					" seconds",
			)
		}

		this.taskId = historyItem ? historyItem.id : (taskId ?? uuidv7())
		this.rootTaskId = historyItem ? historyItem.rootTaskId : rootTask?.taskId
		this.parentTaskId = historyItem ? historyItem.parentTaskId : parentTask?.taskId
		this.childTaskId = undefined

		this.metadata = {
			task: historyItem ? historyItem.task : task,
			images: historyItem ? [] : images,
		}

		// Normal use-case is usually retry similar history task with new workspace.
		this.workspacePath = parentTask
			? parentTask.workspacePath
			: (workspacePath ?? getWorkspacePath(path.join(os.homedir(), "Desktop")))

		this.instanceId = crypto.randomUUID().slice(0, 8)

		this.rooIgnoreController = new RooIgnoreController(this.cwd)
		this.rooProtectedController = new RooProtectedController(this.cwd)
		this.fileContextTracker = new FileContextTracker(host, this.taskId)
		this.cangjieRuntimePolicy = new CangjieRuntimePolicy(this.cwd)

		this.rooIgnoreController.initialize().catch((error) => {
			logger.error("Task", "Failed to initialize RooIgnoreController:", error)
			TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
		})

		this.apiConfiguration = apiConfiguration
		this.api = buildApiHandler(this.apiConfiguration)
		this.autoApprovalHandler = new AutoApprovalHandler()

		this.consecutiveMistakeLimit = consecutiveMistakeLimit ?? DEFAULT_CONSECUTIVE_MISTAKE_LIMIT
		this.hostRef = new WeakRef(host)
		this.globalStoragePath = host.context.globalStorageUri.fsPath
		this.diffViewProvider = host.createDiffViewProvider?.(this.cwd, this) ?? new NullTaskDiffViewProvider()
		this.enableCheckpoints = enableCheckpoints
		this.checkpointTimeout = checkpointTimeout

		this.rootTask = rootTask
		this.parentTask = parentTask
		this.taskNumber = taskNumber
		this.initialStatus = initialStatus
		this.requestBuilder = new TaskRequestBuilder(this)
		this.streamProcessor = new TaskStreamProcessor(this)
		this.errorRecovery = new ErrorRecoveryHandler(this)
		if (parentTask) {
			this.requestBuilder.inheritCacheFromParent(parentTask)
		}

		// Store the task's mode and API config name when it's created.
		// For history items, use the stored values; for new tasks, we'll set them
		// after getting state.
		if (historyItem) {
			this._taskMode = historyItem.mode || defaultModeSlug
			this._taskApiConfigName = historyItem.apiConfigName
			this.taskModeReady = Promise.resolve()
			this.taskApiConfigReady = Promise.resolve()
		} else {
			// For new tasks, don't set the mode/apiConfigName yet - wait for async initialization.
			this._taskMode = undefined
			this._taskApiConfigName = undefined
			this.taskModeReady = this.initializeTaskMode(host)
			this.taskApiConfigReady = this.initializeTaskApiConfigName(host)
		}

		this.assistantMessageParser = undefined

		this.messageQueueService = new MessageQueueService()

		this.messageQueueStateChangedHandler = () => {
			this.emit(NJUST_AI_CJEventName.TaskUserMessage, this.taskId)
			this.emit(NJUST_AI_CJEventName.QueuedMessagesUpdated, this.taskId, this.messageQueueService.messages)
			void this.refreshWebviewState()
		}

		this.messageQueueService.on("stateChanged", this.messageQueueStateChangedHandler)

		// Listen for provider profile changes to update parser state
		this.setupProviderProfileChangeListener(host)

		// Set up diff strategy
		this.diffStrategy = new MultiSearchReplaceDiffStrategy()

		this.toolRepetitionDetector = new ToolRepetitionDetector(this.consecutiveMistakeLimit)
		this.toolExecution.enableAdaptiveTuning()

		// Initialize todo list if provided
		if (initialTodos && initialTodos.length > 0) {
			this.todoList = initialTodos
		}

		// Initialize debounced token usage emit function
		// Uses debounce with maxWait to achieve throttle-like behavior:
		// - leading: true  - Emit immediately on first call
		// - trailing: true - Emit final state when updates stop
		// - maxWait        - Ensures at most one emit per interval during rapid updates (throttle behavior)
		this.debouncedEmitTokenUsage = debounce(
			(tokenUsage: TokenUsage, toolUsage: ToolUsage) => {
				const tokenChanged = hasTokenUsageChanged(tokenUsage, this.tokenUsageSnapshot)
				const toolChanged = hasToolUsageChanged(toolUsage, this.toolUsageSnapshot)

				if (tokenChanged || toolChanged) {
					this.emit(NJUST_AI_CJEventName.TaskTokenUsageUpdated, this.taskId, tokenUsage, toolUsage)
					this.tokenUsageSnapshot = tokenUsage
					this.tokenUsageSnapshotAt = this.clineMessages.at(-1)?.ts
					// Deep copy tool usage for snapshot
					this.toolUsageSnapshot = JSON.parse(JSON.stringify(toolUsage))
				}
			},
			this.TOKEN_USAGE_EMIT_INTERVAL_MS,
			{ leading: true, trailing: true, maxWait: this.TOKEN_USAGE_EMIT_INTERVAL_MS },
		)

		onCreated?.(this)

		if (startTask) {
			this._started = true
			if (task || images) {
				void this.startTask(task, images).catch((error) => {
					logger.error("startTask failed", error)
					TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
				})
			} else if (historyItem) {
				void this.resumeTaskFromHistory().catch((error) => {
					logger.error("resumeTaskFromHistory failed", error)
					TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
				})
			} else {
				throw new Error("Either historyItem or task/images must be provided")
			}
		}
	}

	/**
	 * Initialize the task mode from the provider state.
	 * This method handles async initialization with proper error handling.
	 *
	 * ## Flow
	 * 1. Attempts to fetch the current mode from provider state
	 * 2. Sets `_taskMode` to the fetched mode or `defaultModeSlug` if unavailable
	 * 3. Handles errors gracefully by falling back to default mode
	 * 4. Logs any initialization errors for debugging
	 *
	 * ## Error handling
	 * - Network failures when fetching provider state
	 * - Provider not yet initialized
	 * - Invalid state structure
	 *
	 * All errors result in fallback to `defaultModeSlug` to ensure task can proceed.
	 *
	 * @private
	 * @param provider - The ClineProvider instance to fetch state from
	 * @returns Promise that resolves when initialization is complete
	 */
	private async initializeTaskMode(host: ITaskHost): Promise<void> {
		try {
			const state = await host.getState()
			this._taskMode = state?.mode || defaultModeSlug
		} catch (error) {
			// If there's an error getting state, use the default mode
			this._taskMode = defaultModeSlug
			// Use the provider's log method for better error visibility
			const errorMessage = `Failed to initialize task mode: ${getErrorMessage(error)}`
			host.log(errorMessage)
		}
	}

	/**
	 * Initialize the task API config name from the provider state.
	 * This method handles async initialization with proper error handling.
	 *
	 * ## Flow
	 * 1. Attempts to fetch the current API config name from provider state
	 * 2. Sets `_taskApiConfigName` to the fetched name or "default" if unavailable
	 * 3. Handles errors gracefully by falling back to "default"
	 * 4. Logs any initialization errors for debugging
	 *
	 * ## Error handling
	 * - Network failures when fetching provider state
	 * - Provider not yet initialized
	 * - Invalid state structure
	 *
	 * All errors result in fallback to "default" to ensure task can proceed.
	 *
	 * @private
	 * @param provider - The ClineProvider instance to fetch state from
	 * @returns Promise that resolves when initialization is complete
	 */
	private async initializeTaskApiConfigName(host: ITaskHost): Promise<void> {
		try {
			const state = await host.getState()

			// Avoid clobbering a newer value that may have been set while awaiting provider state
			// (e.g., user switches provider profile immediately after task creation).
			if (this._taskApiConfigName === undefined) {
				this._taskApiConfigName = state?.currentApiConfigName ?? "default"
			}
		} catch (error) {
			// If there's an error getting state, use the default profile (unless a newer value was set).
			if (this._taskApiConfigName === undefined) {
				this._taskApiConfigName = "default"
			}
			// Use the provider's log method for better error visibility
			const errorMessage = `Failed to initialize task API config name: ${getErrorMessage(error)}`
			host.log(errorMessage)
		}
	}

	/**
	 * Sets up a listener for provider profile changes.
	 *
	 * @private
	 * @param provider - The ClineProvider instance to listen to
	 */
	private setupProviderProfileChangeListener(host: ITaskHost): void {
		// Only set up listener if provider has the on method (may not exist in test mocks)
		if (typeof host.on !== "function") {
			return
		}

		this.providerProfileChangeListener = async () => {
			// Abort fetch before awaiting provider state so the in-flight stream stops immediately.
			this.cancelCurrentRequest()
			try {
				const newState = await host.getState()
				if (newState?.apiConfiguration) {
					this.updateApiConfiguration(newState.apiConfiguration)
				}
			} catch (error) {
				logger.error(
					"Task",
					`Failed to update API configuration on profile change for task ${this.taskId}.${this.instanceId}:`,
					error,
				)
				TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
			}
		}

		host.on(NJUST_AI_CJEventName.ProviderProfileChanged, this.providerProfileChangeListener)
	}

	/**
	 * Wait for the task mode to be initialized before proceeding.
	 * This method ensures that any operations depending on the task mode
	 * will have access to the correct mode value.
	 *
	 * ## When to use
	 * - Before accessing mode-specific configurations
	 * - When switching between tasks with different modes
	 * - Before operations that depend on mode-based permissions
	 *
	 * ## Example usage
	 * ```typescript
	 * // Wait for mode initialization before mode-dependent operations
	 * await task.waitForModeInitialization();
	 * const mode = task.taskMode; // Now safe to access synchronously
	 *
	 * // Or use with getTaskMode() for a one-liner
	 * const mode = await task.getTaskMode(); // Internally waits for initialization
	 * ```
	 *
	 * @returns Promise that resolves when the task mode is initialized
	 * @public
	 */
	public async waitForModeInitialization(): Promise<void> {
		return this.taskModeReady
	}

	/**
	 * Get the task mode asynchronously, ensuring it's properly initialized.
	 * This is the recommended way to access the task mode as it guarantees
	 * the mode is available before returning.
	 *
	 * ## Async behavior
	 * - Internally waits for `taskModeReady` promise to resolve
	 * - Returns the initialized mode or `defaultModeSlug` as fallback
	 * - Safe to call multiple times - subsequent calls return immediately if already initialized
	 *
	 * ## Example usage
	 * ```typescript
	 * // Safe async access
	 * const mode = await task.getTaskMode();
	 * console.log(`Task is running in ${mode} mode`);
	 *
	 * // Use in conditional logic
	 * if (await task.getTaskMode() === 'architect') {
	 *   // Perform architect-specific operations
	 * }
	 * ```
	 *
	 * @returns Promise resolving to the task mode string
	 * @public
	 */
	public async getTaskMode(): Promise<string> {
		await this.taskModeReady
		return this._taskMode || defaultModeSlug
	}

	/**
	 * Get the task mode synchronously. This should only be used when you're certain
	 * that the mode has already been initialized (e.g., after waitForModeInitialization).
	 *
	 * ## When to use
	 * - In synchronous contexts where async/await is not available
	 * - After explicitly waiting for initialization via `waitForModeInitialization()`
	 * - In event handlers or callbacks where mode is guaranteed to be initialized
	 *
	 * ## Example usage
	 * ```typescript
	 * // After ensuring initialization
	 * await task.waitForModeInitialization();
	 * const mode = task.taskMode; // Safe synchronous access
	 *
	 * // In an event handler after task is started
	 * task.on('taskStarted', () => {
	 *   console.log(`Task started in ${task.taskMode} mode`); // Safe here
	 * });
	 * ```
	 *
	 * @throws {Error} If the mode hasn't been initialized yet
	 * @returns The task mode string
	 * @public
	 */
	public get taskMode(): string {
		if (this._taskMode === undefined) {
			throw new Error("Task mode accessed before initialization. Use getTaskMode() or wait for taskModeReady.")
		}

		return this._taskMode
	}

	public setTaskMode(mode: string): void {
		this._taskMode = mode
	}

	/**
	 * Wait for the task API config name to be initialized before proceeding.
	 * This method ensures that any operations depending on the task's provider profile
	 * will have access to the correct value.
	 *
	 * ## When to use
	 * - Before accessing provider profile-specific configurations
	 * - When switching between tasks with different provider profiles
	 * - Before operations that depend on the provider profile
	 *
	 * @returns Promise that resolves when the task API config name is initialized
	 * @public
	 */
	public async waitForApiConfigInitialization(): Promise<void> {
		return this.taskApiConfigReady
	}

	/**
	 * Get the task API config name asynchronously, ensuring it's properly initialized.
	 * This is the recommended way to access the task's provider profile as it guarantees
	 * the value is available before returning.
	 *
	 * ## Async behavior
	 * - Internally waits for `taskApiConfigReady` promise to resolve
	 * - Returns the initialized API config name or undefined as fallback
	 * - Safe to call multiple times - subsequent calls return immediately if already initialized
	 *
	 * @returns Promise resolving to the task API config name string or undefined
	 * @public
	 */
	public async getTaskApiConfigName(): Promise<string | undefined> {
		await this.taskApiConfigReady
		return this._taskApiConfigName
	}

	/**
	 * Get the task API config name synchronously. This should only be used when you're certain
	 * that the value has already been initialized (e.g., after waitForApiConfigInitialization).
	 *
	 * ## When to use
	 * - In synchronous contexts where async/await is not available
	 * - After explicitly waiting for initialization via `waitForApiConfigInitialization()`
	 * - In event handlers or callbacks where API config name is guaranteed to be initialized
	 *
	 * Note: Unlike taskMode, this getter does not throw if uninitialized since the API config
	 * name can legitimately be undefined (backward compatibility with tasks created before
	 * this feature was added).
	 *
	 * @returns The task API config name string or undefined
	 * @public
	 */
	public get taskApiConfigName(): string | undefined {
		return this._taskApiConfigName
	}

	/**
	 * Update the task's API config name. This is called when the user switches
	 * provider profiles while a task is active, allowing the task to remember
	 * its new provider profile.
	 *
	 * @param apiConfigName - The new API config name to set
	 * @internal
	 */
	public setTaskApiConfigName(apiConfigName: string | undefined): void {
		this._taskApiConfigName = apiConfigName
	}

	static create(options: TaskOptions): [Task, Promise<void>] {
		const instance = new Task({ ...options, startTask: false })
		const { images, task, historyItem } = options
		let promise

		if (images || task) {
			promise = instance.startTask(task, images)
		} else if (historyItem) {
			promise = instance.resumeTaskFromHistory()
		} else {
			throw new Error("Either historyItem or task/images must be provided")
		}

		return [instance, promise]
	}

	// API Messages

	private async getSavedApiConversationHistory(): Promise<ApiMessage[]> {
		return getSavedApiConversationHistoryWithTask(this.msgMgr)
	}

	async addToApiConversationHistory(message: Anthropic.MessageParam, reasoning?: string) {
		return addToApiConversationHistoryWithTask(this.msgMgr, message, reasoning)
	}

	// NOTE: We intentionally do NOT mutate stored messages to merge consecutive user turns.
	// For API requests, consecutive same-role messages are merged via mergeConsecutiveApiMessages()
	// so rewind/edit behavior can still reference original message boundaries.

	async overwriteApiConversationHistory(newHistory: ApiMessage[]) {
		return overwriteApiConversationHistoryWithTask(this.msgMgr, newHistory)
	}

	/**
	 * Flush any pending tool results to the API conversation history.
	 *
	 * This is critical when the task is about to be
	 * delegated (e.g., via new_task). Before delegation, if other tools were
	 * called in the same turn before new_task, their tool_result blocks are
	 * accumulated in `userMessageContent` but haven't been saved to the API
	 * history yet. If we don't flush them before the parent is disposed,
	 * the API conversation will be incomplete and cause 400 errors when
	 * the parent resumes (missing tool_result for tool_use blocks).
	 *
	 * NOTE: The assistant message is typically already in history by the time
	 * tools execute (added in recursivelyMakeClineRequests after streaming completes).
	 * So we usually only need to flush the pending user message with tool_results.
	 */
	public async flushPendingToolResultsToHistory(): Promise<boolean> {
		return flushPendingToolResultsToHistoryWithTask(this.msgMgr)
	}

	private async saveApiConversationHistory(): Promise<boolean> {
		return saveApiConversationHistoryWithTask(this.msgMgr)
	}

	/**
	 * Public wrapper to retry saving the API conversation history.
	 * Uses exponential backoff: up to 3 attempts with delays of 100 ms, 500 ms, 1500 ms.
	 * Used by delegation flow when flushPendingToolResultsToHistory reports failure.
	 */
	public async retrySaveApiConversationHistory(): Promise<boolean> {
		return retrySaveApiConversationHistoryWithTask(this.msgMgr)
	}

	// Cline Messages

	private async getSavedClineMessages(): Promise<ClineMessage[]> {
		return getSavedClineMessagesWithTask(this.msgMgr)
	}

	private async addToClineMessages(message: Omit<ClineMessage, "id"> & { id?: string }) {
		return addToClineMessagesWithTask(this.msgMgr, message)
	}

	public async overwriteClineMessages(newMessages: ClineMessage[]) {
		return overwriteClineMessagesWithTask(this.msgMgr, newMessages)
	}

	private async updateClineMessage(message: ClineMessage) {
		return updateClineMessageWithTask(this.msgMgr, message)
	}

	private async saveClineMessages(): Promise<boolean> {
		return saveClineMessagesWithTask(this.msgMgr)
	}

	/** @deprecated Prefer findMessageById for new code. */
	private findMessageByTimestamp(ts: number): ClineMessage | undefined {
		return findMessageByTimestampWithTask(this.msgMgr, ts)
	}

	private findMessageById(id: string): ClineMessage | undefined {
		return findMessageByIdWithTask(this.msgMgr, id)
	}

	// Note that `partial` has three valid states true (partial message),
	// false (completion of partial message), undefined (individual complete
	// message).
	async ask(
		type: ClineAsk,
		text?: string,
		partial?: boolean,
		progressStatus?: ToolProgressStatus,
		isProtected?: boolean,
	): Promise<{ response: ClineAskResponse; text?: string; images?: string[] }> {
		return this.askSayHandler.ask(type, text, partial, progressStatus, isProtected)
	}

	handleWebviewAskResponse(askResponse: ClineAskResponse, text?: string, images?: string[]) {
		// Clear any pending auto-approval timeout when user responds
		this.cancelAutoApprovalTimeout()

		this.askResponse = askResponse
		this.askResponseText = text
		this.askResponseImages = images

		// Create a checkpoint whenever the user sends a message.
		// Use allowEmpty=true to ensure a checkpoint is recorded even if there are no file changes.
		// Suppress the checkpoint_saved chat row for this particular checkpoint to keep the timeline clean.
		if (askResponse === "messageResponse") {
			void this.checkpointSave(false, true)
		}

		// Mark the last follow-up question as answered
		if (askResponse === "messageResponse" || askResponse === "yesButtonClicked") {
			// Find the last unanswered follow-up message using findLastIndex
			const lastFollowUpIndex = findLastIndex(
				this.clineMessages,
				(msg) => msg.type === "ask" && msg.ask === "followup" && !msg.isAnswered,
			)

			if (lastFollowUpIndex !== -1) {
				// Mark this follow-up as answered
				this.clineMessages[lastFollowUpIndex]!.isAnswered = true
				// Save the updated messages
				this.saveClineMessages().catch((error) => {
					logger.error("Task", "Failed to save answered follow-up state:", error)
					TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
				})
			}
		}

		// Mark the last tool-approval ask as answered when user approves (or auto-approval)
		if (askResponse === "yesButtonClicked") {
			const lastToolAskIndex = findLastIndex(
				this.clineMessages,
				(msg) => msg.type === "ask" && msg.ask === "tool" && !msg.isAnswered,
			)
			if (lastToolAskIndex !== -1) {
				this.clineMessages[lastToolAskIndex]!.isAnswered = true
				void this.updateClineMessage(this.clineMessages[lastToolAskIndex]!).catch((error) => {
					logger.warn("Task", "updateClineMessage failed", error)
				})
				this.saveClineMessages().catch((error) => {
					logger.error("Task", "Failed to save answered tool-ask state:", error)
					TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
				})
			}
		}
	}

	/**
	 * Cancel any pending auto-approval timeout.
	 * Called when user interacts (types, clicks buttons, etc.) to prevent the timeout from firing.
	 */
	public cancelAutoApprovalTimeout(): void {
		if (this.autoApprovalTimeoutRef) {
			clearTimeout(this.autoApprovalTimeoutRef)
			this.autoApprovalTimeoutRef = undefined
		}
	}

	public approveAsk({ text, images }: { text?: string; images?: string[] } = {}) {
		this.handleWebviewAskResponse("yesButtonClicked", text, images)
	}

	public denyAsk({ text, images }: { text?: string; images?: string[] } = {}) {
		this.handleWebviewAskResponse("noButtonClicked", text, images)
	}

	public supersedePendingAsk(): void {
		this.lastMessageTs = Date.now()
	}

	/**
	 * Updates the API configuration and rebuilds the API handler.
	 * Cancels any in-flight request and resets streaming/parser/token-cache state so
	 * mixed dimensions cannot leak across profiles or models.
	 *
	 * @param newApiConfiguration - The new API configuration to use
	 */
	public updateApiConfiguration(newApiConfiguration: ProviderSettings): void {
		this.cancelCurrentRequest()
		try {
			this.toolCallParser.clearAllStreamingToolCalls()
			this.toolCallParser.clearRawChunkState()
		} catch (err) {
			logger.error("Task", "Failed to clear streaming tool call state:", err)
		}
		this.debouncedEmitTokenUsage.cancel()

		this.isWaitingForFirstChunk = false
		this.isStreaming = false
		this.currentStreamingContentIndex = 0
		this.currentStreamingDidCheckpoint = false
		this.assistantMessageContent = []
		this.presentAssistantMessageLocked = false
		this.presentAssistantMessageHasPendingUpdates = false
		this.didCompleteReadingStream = false
		this.userMessageContent = []
		this.userMessageContentReady = false
		this.didRejectTool = false
		this.didAlreadyUseTool = false
		this.didToolFailInCurrentTurn = false
		this.assistantMessageSavedToHistory = false
		this.streamingToolCallIndices.clear()
		this.cachedStreamingModel = undefined
		this.tokenUsageSnapshot = undefined
		this.tokenUsageSnapshotAt = undefined
		this.toolUsageSnapshot = undefined
		tokenCountCache.clear()

		this.apiConfiguration = newApiConfiguration
		this.api = buildApiHandler(this.apiConfiguration)
	}

	public async submitUserMessage(
		text: string,
		images?: string[],
		mode?: string,
		providerProfile?: string,
	): Promise<void> {
		try {
			text = (text ?? "").trim()
			images = images ?? []

			if (text.length === 0 && images.length === 0) {
				return
			}

			const provider = this.hostRef.deref()

			if (provider) {
				if (mode) {
					await provider.setMode(mode)
				}

				if (providerProfile) {
					await provider.setProviderProfile(providerProfile)

					// Update this task's API configuration to match the new profile
					// This ensures the parser state is synchronized with the selected model
					const newState = await provider.getState()
					if (newState?.apiConfiguration) {
						this.updateApiConfiguration(newState.apiConfiguration)
					}
				}

				this.emit(NJUST_AI_CJEventName.TaskUserMessage, this.taskId)

				// Handle the message directly instead of routing through the webview.
				// This avoids a race condition where the webview's message state hasn't
				// hydrated yet, causing it to interpret the message as a new task request.
				this.handleWebviewAskResponse("messageResponse", text, images)
			} else {
				logger.error("Task", "submitUserMessage: Provider reference lost")
			}
		} catch (error) {
			logger.error("Task", "submitUserMessage: Failed to submit user message:", error)
			TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
		}
	}

	handleTerminalOperation(terminalOperation: "continue" | "abort") {
		if (terminalOperation === "continue") {
			this.terminalProcess?.continue()
		} else if (terminalOperation === "abort") {
			this.terminalProcess?.abort()
		}
	}

	// Delegated to TaskStreamProcessor
	private async getFilesReadByRooSafely(context: string): Promise<string[] | undefined> {
		return this.streamProcessor.getFilesReadByRooSafely(context)
	}

	public async condenseContext(): Promise<void> {
		return this.requestBuilder.condenseContext()
	}

	async say(
		type: ClineSay,
		text?: string,
		images?: string[],
		partial?: boolean,
		checkpoint?: Record<string, UnsafeAny>,
		progressStatus?: ToolProgressStatus,
		options: {
			isNonInteractive?: boolean
		} = {},
		contextCondense?: ContextCondense,
		contextTruncation?: ContextTruncation,
	): Promise<undefined> {
		return this.askSayHandler.say(
			type,
			text,
			images,
			partial,
			checkpoint,
			progressStatus,
			options,
			contextCondense,
			contextTruncation,
		)
	}

	async sayAndCreateMissingParamError(toolName: ToolName, paramName: string, relPath?: string) {
		return this.askSayHandler.sayAndCreateMissingParamError(toolName, paramName, relPath)
	}

	// Lifecycle
	// Start / Resume / Abort / Dispose

	/**
	 * Get enabled MCP tools count for this task.
	 * Returns the count along with the number of servers contributing.
	 *
	 * @returns Object with enabledToolCount and enabledServerCount
	 */
	private async getEnabledMcpToolsCount(): Promise<{ enabledToolCount: number; enabledServerCount: number }> {
		try {
			const provider = this.hostRef.deref()
			if (!provider) {
				return { enabledToolCount: 0, enabledServerCount: 0 }
			}

			const { mcpEnabled } = await provider.getState()
			if (!(mcpEnabled ?? true)) {
				return { enabledToolCount: 0, enabledServerCount: 0 }
			}

			const mcpHub = provider.getMcpHub()
			if (!mcpHub) {
				return { enabledToolCount: 0, enabledServerCount: 0 }
			}

			const servers = mcpHub.getServers()
			return countEnabledMcpTools(servers)
		} catch (error) {
			logger.error("Task", "getEnabledMcpToolsCount: Error counting MCP tools:", error)
			TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
			return { enabledToolCount: 0, enabledServerCount: 0 }
		}
	}

	/**
	 * Manually start a **new** task when it was created with `startTask: false`.
	 *
	 * This fires `startTask` as a background async operation for the
	 * `task/images` code-path only.  It does **not** handle the
	 * `historyItem` resume path (use the constructor with `startTask: true`
	 * for that).  The primary use-case is in the delegation flow where the
	 * parent's metadata must be persisted to globalState **before** the
	 * child task begins writing its own history (avoiding a read-modify-write
	 * race on globalState).
	 */
	public start(): void {
		if (this._started) {
			return
		}
		this._started = true

		const { task, images } = this.metadata

		if (task || images) {
			void this.startTask(task ?? undefined, images ?? undefined).catch((error) => {
				logger.error("startTask failed", error)
				TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
			})
		}
	}

	private async startTask(task?: string, images?: string[]): Promise<void> {
		return this.lifecycleHandler.startTask(task, images)
	}

	private async resumeTaskFromHistory() {
		return this.lifecycleHandler.resumeTaskFromHistory()
	}

	/**
	 * Cancels the current HTTP request if one is in progress.
	 * This immediately aborts the underlying stream rather than waiting for the next chunk.
	 */
	public cancelCurrentRequest(): void {
		if (this.currentRequestAbortController) {
			logger.info("Task", `Aborting current HTTP request for task ${this.taskId}.${this.instanceId}`)
			this.currentRequestAbortController.abort()
			this.currentRequestAbortController = undefined
		}
	}

	/**
	 * Force emit a final token usage update, ignoring throttle.
	 * Called before task completion or abort to ensure final stats are captured.
	 * Triggers the debounce with current values and immediately flushes to ensure emit.
	 */
	public emitFinalTokenUsageUpdate(): void {
		const tokenUsage = this.getTokenUsage()
		this.debouncedEmitTokenUsage(tokenUsage, this.toolUsage)
		this.debouncedEmitTokenUsage.flush()
	}

	/** Signal the outer loop that the task has reached a terminal completed
	 *  state via attempt_completion acceptance. */
	public markTaskCompleted(): void {
		this.taskCompleted = true
	}

	/** Return a Promise that resolves when background switch is requested */
	getBackgroundSignal(): Promise<void> {
		if (!this._backgroundSignal) {
			this._backgroundSignal = new Promise<void>((resolve) => {
				this._backgroundResolve = resolve
			})
		}
		return this._backgroundSignal
	}

	/** Request this task be moved to background. Resolves the signal promise. */
	requestBackground(): void {
		if (this._backgroundResolve && !this.isBackgrounded) {
			this.isBackgrounded = true
			this._backgroundResolve()
			this._backgroundResolve = null
		}
	}

	/** Reset background state (call when task completes or is reused) */
	private _resetBackgroundState(): void {
		this._backgroundSignal = null
		this._backgroundResolve = null
		this.isBackgrounded = false
	}
	public async abortTask(isAbandoned = false) {
		return this.lifecycleHandler.abortTask(isAbandoned)
	}

	public isDisposed = false

	public dispose(): void {
		return this.lifecycleHandler.dispose()
	}

	// Subtasks
	// Spawn / Wait / Complete

	public async startSubtask(
		message: string,
		initialTodos: TodoItem[],
		mode: string,
		isolationLevel: IsolationLevel = "shared",
		forkedConfig?: ForkedContextConfig,
		cacheSafeParams?: CacheSafeParams,
	) {
		return this.subtaskHandler.startSubtask(
			message,
			initialTodos,
			mode,
			isolationLevel,
			forkedConfig,
			cacheSafeParams,
		)
	}

	/**
	 * Resume parent task after delegation completion without showing resume ask.
	 * Used in metadata-driven subtask flow.
	 *
	 * This method:
	 * - Clears any pending ask states
	 * - Resets abort and streaming flags
	 * - Ensures next API call includes full context
	 * - Immediately continues task loop without user interaction
	 */
	public async resumeAfterDelegation(): Promise<void> {
		return this.subtaskHandler.resumeAfterDelegation()
	}

	// Cloud Agent orchestration delegated to CloudAgentOrchestrator

	private async initiateCloudAgentLoop(userMessage: string, images?: string[]): Promise<void> {
		const host: ICloudAgentHost = {
			taskId: this.taskId,
			cwd: this.cwd,
			get abort() {
				return (this as UnsafeAny as { _task: Task })._task?.abort ?? false
			},
			rooIgnoreController: this.rooIgnoreController,
			rooProtectedController: this.rooProtectedController,
			say: (type, text?, imgs?) => this.say(type, text, imgs),
			ask: (type, text?, partial?) => this.ask(type, text, partial),
			emit: (event, ...args) => EventEmitter.prototype.emit.call(this, event, ...args) as boolean,
			setCurrentRequestAbortController: (ctrl) => {
				this.currentRequestAbortController = ctrl
			},
			compileLocal: async (cwd) => {
				const provider = this.hostRef.deref()
				if (!provider?.compileLocal) {
					throw new Error("本地编译功能未配置，请确认 Cangjie SDK 已安装。")
				}
				return provider.compileLocal(cwd)
			},
		}
		// Patch the abort getter to actually read from this task instance
		Object.defineProperty(host, "abort", { get: () => this.abort })

		const { CloudAgentOrchestrator } = await import("./CloudAgentOrchestrator")
		const orchestrator = new CloudAgentOrchestrator(host)
		await orchestrator.run(userMessage, images)
	}

	// Task Loop

	private async initiateTaskLoop(userContent: Anthropic.Messages.ContentBlockParam[]): Promise<void> {
		// Kicks off the checkpoints initialization process in the background.
		void getCheckpointService(this).catch((error) => {
			logger.warn("Task", "getCheckpointService failed", error)
			TelemetryService.reportError(error, TelemetryEventName.TASK_LIFECYCLE_ERROR)
		})

		// Start skill/memory prefetch in parallel (non-blocking)
		const provider = this.hostRef.deref()
		startAllPrefetch({
			// eslint-disable-next-line @typescript-eslint/require-await
			skillFetchFn: async () => {
				const skillsManager = provider?.getSkillsManager()
				const skills = skillsManager?.getAllSkills() ?? []
				return skills.map((s) => s.name)
			},
		})

		// MemRL: inject dependencies then retrieve episodic hints and LTM rules
		// Pass this.cwd explicitly so the correct workspace path is always used
		const memoryManager = provider?.getMemoryManager(this.cwd)
		if (memoryManager) {
			// Try to get embedder from CodeIndexManager (best-effort, may be undefined)
			const embedder = provider && "getCurrentWorkspaceCodeIndexManager" in provider
				? (provider as import("../../core/webview/ClineProvider").ClineProvider)
					.getCurrentWorkspaceCodeIndexManager()
					?.tryCreateEmbedder()
				: undefined
			memoryManager.updateDependencies(this.api, embedder)

			const intent = this.getTaskIntent()
			try {
				const { episodicHints, ltmRules } = await memoryManager.beforeRun(this.taskId, intent)
				this.memrlEpisodicHints = episodicHints
				this.memrlLtmRules = ltmRules
				// Invalidate system prompt cache so new hints are injected
				this.requestBuilder["systemPromptPartsCache"] = undefined
			} catch {
				// Non-blocking: failures must not prevent task execution
			}
		}

		let nextUserContent = userContent
		let includeFileDetails = true

		this.emit(NJUST_AI_CJEventName.TaskStarted)

		while (!this.abort && !this.taskCompleted) {
			const didEndLoop = await this.recursivelyMakeClineRequests(nextUserContent, includeFileDetails)
			includeFileDetails = false

			if (didEndLoop) {
				// Only happens when max requests is hit and user denies
				// resetting the count, or an unexpected error is caught.
				break
			}

			if (this.taskCompleted) {
				// attempt_completion was accepted — stop without
				// re-prompting the model.
				break
			}

			nextUserContent = [{ type: "text", text: formatResponse.noToolsUsed() }]
		}

		// MemRL: afterRun — fire-and-forget, must not block task completion
		if (memoryManager) {
			const intent = this.getTaskIntent()
			const stm = memoryManager.getStm(this.taskId)
			const stmSummary = stm.summarize()
			// Reward: 1.0 on successful completion, 0.0 on abort
			const reward = this.taskCompleted ? 1.0 : 0.0
			memoryManager.afterRun(this.taskId, intent, stmSummary, reward)
		}
	}

	/** Extract a plain-text task intent from the initial user content for MemRL. */
	private getTaskIntent(): string {
		const history = this.apiConversationHistory
		for (let i = 0; i < history.length; i++) {
			const m = history[i] as { role: string; content: string | Array<{ type: string; text?: string }> }
			if (m.role !== "user") continue
			if (typeof m.content === "string") return m.content.slice(0, 500)
			if (Array.isArray(m.content)) {
				for (const block of m.content) {
					if (block.type === "text" && block.text) return block.text.slice(0, 500)
				}
			}
		}
		return this.taskId
	}

	public async recursivelyMakeClineRequests(
		userContent: Anthropic.Messages.ContentBlockParam[],
		includeFileDetails: boolean = false,
	): Promise<boolean> {
		return this.executor.recursivelyMakeClineRequests(userContent, includeFileDetails)
	}

	private async getSystemPromptParts(): Promise<SystemPromptParts> {
		return this.requestBuilder.getSystemPromptParts()
	}

	private async getSystemPrompt(): Promise<string> {
		return this.requestBuilder.getSystemPrompt()
	}

	// Delegated to TaskStreamProcessor
	private getCurrentProfileId(state: UnsafeAny): string {
		return this.streamProcessor.getCurrentProfileId(state)
	}

	// Delegated to TaskStreamProcessor
	async handleContextWindowExceededError(): Promise<void> {
		return this.streamProcessor.handleContextWindowExceededError()
	}

	// Delegated to TaskStreamProcessor
	private async maybeWaitForProviderRateLimit(retryAttempt: number): Promise<void> {
		return this.streamProcessor.maybeWaitForProviderRateLimit(retryAttempt)
	}

	public async *attemptApiRequest(
		retryAttempt: number = 0,
		options: { skipProviderRateLimit?: boolean } = {},
	): ApiStream {
		yield* this.executor.attemptApiRequest(retryAttempt, options)
	}

	// Delegated to TaskStreamProcessor
	// Shared exponential backoff for retries (first-chunk and mid-stream)
	private async backoffAndAnnounce(retryAttempt: number, error: UnsafeAny): Promise<void> {
		return this.streamProcessor.backoffAndAnnounce(retryAttempt, error)
	}

	// Checkpoints

	public async checkpointSave(force: boolean = false, suppressMessage: boolean = false) {
		return checkpointSave(this, force, suppressMessage)
	}

	// Delegated to TaskStreamProcessor
	private buildCleanConversationHistory(
		messages: ApiMessage[],
	): Array<
		| Anthropic.Messages.MessageParam
		| { type: "reasoning"; encrypted_content?: string; id?: string; summary?: UnsafeAny[] }
	> {
		return this.streamProcessor.buildCleanConversationHistory(messages)
	}
	public async checkpointRestore(options: CheckpointRestoreOptions) {
		return checkpointRestore(this, options)
	}

	public async checkpointDiff(options: CheckpointDiffOptions) {
		return checkpointDiff(this, options)
	}

	// Metrics

	public combineMessages(messages: ClineMessage[]) {
		return combineApiRequests(combineCommandSequences(messages))
	}

	public getTokenUsage(): TokenUsage {
		return getApiMetrics(this.combineMessages(this.clineMessages.slice(1)))
	}

	// checkSubtaskTokenBudget() moved to TaskExecutor

	public recordToolUsage(toolName: ToolName) {
		if (!this.toolUsage[toolName]) {
			this.toolUsage[toolName] = { attempts: 0, failures: 0 }
		}

		this.toolUsage[toolName].attempts++
	}

	public recordToolError(toolName: ToolName, error?: string) {
		if (!this.toolUsage[toolName]) {
			this.toolUsage[toolName] = { attempts: 0, failures: 0 }
		}

		this.toolUsage[toolName].failures++
		this.toolExecution.recordToolErrorMetric(toolName)

		if (error) {
			this.emit(NJUST_AI_CJEventName.TaskToolFailed, this.taskId, toolName, error)
		}
	}

	// Getters

	public get taskStatus(): TaskStatus {
		if (this.interactiveAsk) {
			return TaskStatus.Interactive
		}

		if (this.resumableAsk) {
			return TaskStatus.Resumable
		}

		if (this.idleAsk) {
			return TaskStatus.Idle
		}

		return TaskStatus.Running
	}

	public get taskAsk(): ClineMessage | undefined {
		return this.idleAsk || this.resumableAsk || this.interactiveAsk
	}

	public get queuedMessages(): QueuedMessage[] {
		return this.messageQueueService.messages
	}

	public get tokenUsage(): TokenUsage | undefined {
		if (this.tokenUsageSnapshot && this.tokenUsageSnapshotAt) {
			return this.tokenUsageSnapshot
		}

		this.tokenUsageSnapshot = this.getTokenUsage()
		this.tokenUsageSnapshotAt = this.clineMessages.at(-1)?.ts

		return this.tokenUsageSnapshot
	}

	public get cwd() {
		return this.workspacePath
	}

	/**
	 * Provides convenient access to high-level message operations.
	 * Uses lazy initialization - the MessageManager is only created when first accessed.
	 * Subsequent accesses return the same cached instance.
	 *
	 * ## Important: Single Coordination Point
	 *
	 * **All MessageManager operations must go through this getter** rather than
	 * instantiating `new MessageManager(task)` directly. This ensures:
	 * - A single shared instance for consistent behavior
	 * - Centralized coordination of all rewind/message operations
	 * - Ability to add internal state or instrumentation in the future
	 *
	 * @example
	 * ```typescript
	 * // Correct: Use the getter
	 * await task.messageManager.rewindToTimestamp(ts)
	 *
	 * // Incorrect: Do NOT create new instances directly
	 * // const manager = new MessageManager(task) // Don't do this!
	 * ```
	 */
	get messageManager(): MessageManager {
		if (!this._messageManager) {
			this._messageManager = new MessageManager(this)
		}
		return this._messageManager
	}

	/**
	 * Process any queued messages by dequeuing and submitting them.
	 * This ensures that queued user messages are sent when appropriate,
	 * preventing them from getting stuck in the queue.
	 *
	 * @param context - Context string for logging (e.g., the calling tool name)
	 */
	public processQueuedMessages(): void {
		try {
			if (!this.messageQueueService.isEmpty()) {
				const queued = this.messageQueueService.dequeueMessage()
				if (queued) {
				this.queuedMessageTimer = setTimeout(() => {
					this.queuedMessageTimer = undefined
					this.submitUserMessage(queued.text, queued.images).catch((err) => {
						logger.error("Task", "Failed to submit queued message:", err)
						TelemetryService.reportError(err, TelemetryEventName.TASK_LIFECYCLE_ERROR)
					})
				}, 0)
				}
			}
		} catch (e) {
			logger.error("Task", "Queue processing error:", e)
			TelemetryService.reportError(e, TelemetryEventName.TASK_LIFECYCLE_ERROR)
		}
	}
}
