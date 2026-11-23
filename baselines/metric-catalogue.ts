export type ProjectMetric = {
  id: string;
  name: string;
  unit: string;
  definition: string;
  importance: string;
  group: string;
};

const outcomes = 'Research outcomes';
const resources = 'GPU use and service behavior';
const diagnostics = 'Drafting and delivery diagnostics';

export const projectMetricCatalogue: ProjectMetric[] = [
  {
    id: 'metric-otps',
    name: 'Output tokens per second (OTPS)',
    unit: 'output tokens/s',
    definition: 'Add the output tokens that all requests receive, including partial responses. Divide the total by the measured interval through the final completion or timeout. Count output tokens after the stopping and output-length rules apply.',
    importance: 'OTPS measures the total output delivered each second. A common token count and timing interval support comparisons across methods. P-EAGLE calls this aggregate rate OTPS. DFlash also uses throughput or TPS.',
    group: outcomes,
  },
  {
    id: 'metric-gpu-time',
    name: 'Allocated GPU time per million output tokens',
    unit: 'GPU-seconds/million output tokens',
    definition: 'Multiply the benchmark’s elapsed seconds by the number of allocated GPUs. Divide by its delivered output tokens and multiply by one million. The comparison tables cover each timed benchmark block. Full-job accounting also includes model loading, warmups and idle time outside those blocks.',
    importance: 'GPU time shows how much reserved computing capacity a method needs to deliver a fixed amount of output. A lower value means lower compute cost for the recorded outputs.',
    group: outcomes,
  },
  {
    id: 'metric-quality',
    name: 'Exact output agreement with AR',
    unit: '% of checked requests',
    definition: 'Under greedy decoding, compare each method with the target model generating alone in the same engine. Match the prompt and generation settings. Count requests with identical output token IDs, output lengths and stopping reasons. Divide by all checked requests and multiply by 100.',
    importance: 'Output agreement checks whether acceleration preserves the baseline answer on the checked prompts. Separate task scores, such as math accuracy and code pass@1, assess whether the answers solve the tasks. Report failures and timeouts alongside agreement.',
    group: outcomes,
  },
  {
    id: 'metric-speedup',
    name: 'Speedup over AR',
    unit: '×',
    definition: 'Divide the method’s OTPS by the OTPS of autoregressive decoding in the matching engine. Match the prompts, output rules, hardware and request schedule. A ratio of 1.2 means the method delivers 20% more output per second.',
    importance: 'The ratio measures acceleration over the target model generating alone. P-EAGLE’s published comparison against sequential EAGLE-3 retains its own reference label. Source reproductions also retain each paper’s timing and averaging rules.',
    group: outcomes,
  },
  {
    id: 'metric-completion',
    name: 'End-to-end request latency',
    unit: 'ms/request, median and p95',
    definition: 'Time each successful request from actual submission until completion. Sort those times and report the median and the 95th percentile (p95). The median describes the middle request. The p95 describes the time within which roughly 95% of successful requests finish.',
    importance: 'End-to-end request latency measures how long a user waits for the complete answer. The median and p95 describe the typical request and slower requests.',
    group: outcomes,
  },
  {
    id: 'metric-ttft',
    name: 'Time to first token (TTFT)',
    unit: 'ms/request, median and p95',
    definition: 'Time each request from actual client submission until the first streamed output arrives. Use the first nonempty output chunk as the client observation. Exclude metadata-only events. Report the median and p95 among requests that return output, and count requests that return none.',
    importance: 'TTFT measures the initial wait for an answer. A streamed chunk can contain several tokens, so the recorded time describes delivery to the client. Record buffering and streaming settings with the result.',
    group: outcomes,
  },
  {
    id: 'metric-pause',
    name: 'Maximum inter-chunk latency per request',
    unit: 'ms, p95 across eligible requests',
    definition: 'A streaming server sends an answer in pieces called chunks. For each request with at least two nonempty chunks, find the longest gap between successive chunks. Take the p95 of those longest gaps, giving each eligible request one observation.',
    importance: 'This project measure captures long interruptions while people read an answer. Each eligible request contributes one observation to the p95.',
    group: outcomes,
  },
  {
    id: 'metric-memory',
    name: 'Peak GPU memory',
    unit: 'GiB/device',
    definition: 'Record the highest observed memory use on each GPU during the stated interval. Keep separate readings for memory used on the whole device, memory occupied by live tensors and memory reserved by the framework. A gibibyte (GiB) equals 1,073,741,824 bytes.',
    importance: 'Peak memory shows which GPU can hold the target model, drafter and working data together. Memory needs can affect how many devices the project must reserve.',
    group: resources,
  },
  {
    id: 'metric-per-gpu-throughput',
    name: 'Output throughput per GPU',
    unit: 'output tokens/s/GPU',
    definition: 'Divide a benchmark’s delivered output tokens by its elapsed seconds multiplied by the allocated GPU count. The comparison tables use timed benchmark blocks. Complete-job resource totals retain their own allocation interval.',
    importance: 'This rate measures output relative to allocated GPU time. DSpark reports related production throughput per GPU. Record the GPU model, allocation interval and GPU count for each project result.',
    group: resources,
  },
  {
    id: 'metric-request-speed',
    name: 'End-to-end output rate per request',
    unit: 'output tokens/s/request',
    definition: 'For each completed request, divide its returned output tokens by the seconds from actual submission through completion. Average the request rates equally. The clock includes waiting, prompt processing and generation.',
    importance: 'End-to-end output rate per request describes the complete response speed experienced by individual users. DSpark calls its per-user generation speed TPS. Its production timer and averaging rule require source-specific evidence.',
    group: resources,
  },
  {
    id: 'metric-tpot',
    name: 'Time per output token (TPOT)',
    unit: 'ms/output token',
    definition: 'For each completed response with N output tokens, require N greater than one. Subtract TTFT from end-to-end request latency and divide by N minus one. Average the eligible request values equally. Count excluded responses separately.',
    importance: 'TPOT measures average delivery time after the first streamed output through request completion. The client observes chunks that can contain several tokens. Record buffering and completion events with the result.',
    group: diagnostics,
  },
  {
    id: 'metric-pooled-gap',
    name: 'Inter-chunk latency',
    unit: 'ms/gap, p95',
    definition: 'Within each response, measure the gaps between successive nonempty chunks. Pool all observed gaps and report their p95. A response with more chunks contributes more gaps.',
    importance: 'This measure describes the long pauses among individual deliveries. The separate longest-pause measure gives each eligible response equal weight.',
    group: diagnostics,
  },
  {
    id: 'metric-draft-fraction',
    name: 'Draft-token acceptance rate',
    unit: '% of proposed draft tokens',
    definition: 'Divide accepted draft tokens by proposed draft tokens and multiply by 100. The numerator and denominator count draft tokens only. Record target correction and bonus tokens separately. For a branching draft, state which branches the proposal count covers.',
    importance: 'Draft-token acceptance rate measures how much proposed output passes verification. DSpark generally includes a target bonus in its published acceptance statistics. Each source comparison preserves the paper’s numerator and denominator.',
    group: diagnostics,
  },
  {
    id: 'metric-accepted-round',
    name: 'Accepted draft tokens per round',
    unit: 'accepted draft tokens/round',
    definition: 'Divide the total accepted draft tokens by the number of verification rounds. Include rounds that accept zero draft tokens. Record tokens supplied by the target separately.',
    importance: 'This average shows how many proposed tokens each target check accepts. Reading it with round time helps explain differences in speed.',
    group: diagnostics,
  },
  {
    id: 'metric-acceptance-length',
    name: 'Mean acceptance length (τ)',
    unit: 'committed tokens/round',
    definition: 'Add the tokens retained across all verification rounds before final output trimming. Include accepted draft tokens and any target correction or bonus tokens that the decoder retains. Divide by the total round count, giving each round equal weight. Record initial tokens generated outside verification rounds separately.',
    importance: 'Mean acceptance length measures progress per target check. The symbol τ is pronounced tau. Longer responses contribute more rounds. The project definition pools rounds. Source reproductions retain the paper’s stopping and averaging rules.',
    group: outcomes,
  },
  {
    id: 'metric-acceptance-length-request',
    name: 'Mean acceptance length (τ), averaged across requests',
    unit: 'committed tokens/round',
    definition: 'For each request with at least one verification round, divide its retained tokens by its round count. Include accepted drafts and retained target tokens before final output trimming. Average the request values equally.',
    importance: 'This supporting diagnostic gives each eligible request one observation. The current DFlash evaluator also averages request means, with final output trimming applied before counting. A DFlash reproduction preserves both of those source rules.',
    group: diagnostics,
  },
  {
    id: 'metric-returned-round',
    name: 'Delivered output tokens per round',
    unit: 'delivered tokens/round',
    definition: 'Count the output tokens from the recorded rounds that actually reach the client after stopping and output limits. Divide by all those rounds, including rounds that return zero tokens. Record any initial target token outside the rounds separately.',
    importance: 'This average shows how much useful output a round delivers. Reading it with time per round helps explain the observed OTPS.',
    group: diagnostics,
  },
  {
    id: 'metric-prefix-survival',
    name: 'Prefix acceptance probability by position',
    unit: 'probability at each draft position',
    definition: 'For each draft position, count the rounds that proposed that position. Among those rounds, count how often the target accepted every draft token through that position before final output clipping. Divide the second count by the first.',
    importance: 'The position-by-position results show where proposed tokens begin to fail verification. They can guide a later test of how many positions the drafter should propose.',
    group: diagnostics,
  },
  {
    id: 'metric-conditional-acceptance',
    name: 'Conditional acceptance rate by position',
    unit: '% at each draft position',
    definition: 'At each draft position, count rounds that proposed the position and accepted every earlier proposal. Among these eligible rounds, count how often the current proposal passes. Divide passes by eligible rounds and multiply by 100. Report a result only when the eligible count is positive.',
    importance: 'Conditional acceptance identifies how reliably a proposal passes after its earlier tokens succeed. EAGLE-3’s n-α and DSpark’s positional analysis use this conditional question. Each source comparison also preserves its feature-history and position conventions.',
    group: diagnostics,
  },
  {
    id: 'metric-component-time',
    name: 'Draft latency, verification latency and round latency',
    unit: 'ms/round for each component',
    definition: 'In a separate timing trace, measure a full cycle from drafting through verification and the final state update. Measure the time spent drafting, checking and moving data within that cycle. Record which operations overlap and average times over the observed cycles.',
    importance: 'Comparing component times reveals which operation takes longest and helps identify changes that could improve response speed or reduce GPU use.',
    group: diagnostics,
  },
];
