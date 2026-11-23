import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const templatePath = process.env.HPML_TEMPLATE ?? '/Users/ayo/Downloads/HPML_Midpoint_Report_Template.pptx';
const outputDir = process.env.HPML_DECK_BUILD ?? '/private/tmp/hpml-category-deck/build';
const screenshotPaths = process.argv.slice(2);
if(screenshotPaths.length!==6) throw new Error('Supply six actual Overleaf panel screenshots: mathematics OTPS/GPU time, code OTPS/GPU time, and conversation/summarization OTPS/GPU time.');
const deck = await PresentationFile.importPptx(await FileBlob.load(templatePath));
const original = [...deck.slides.items];
const reportSource = 'Source: PROJECT/hpml-project/measurements/report/report.tex and its generated result tables. The saved publication contains 810 complete timed responses across nine configurations. Each configuration uses 30 selected prompts and three repetitions.';
const prose = [];

function setTitle(slide, text, size='44pt') {
  const shape = slide.shapes.items[0];
  shape.text = text;
  shape.text.style = {typeface:'Calibri Light',fontSize:size,color:'#000000',autoFit:'none'};
  prose.push(text);
}
function sections(slide, items, size='20pt') {
  const shape=slide.shapes.items[1];
  shape.text=items.length===6 ? items.map(([heading,body,uri])=>({
    bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:1000,
    runs:[{run:heading+'. ',style:{bold:true}},{run:body,...(uri?{link:{uri,isExternal:true},style:{color:'#000000',underline:'sng'}}:{})}]
  })) : items.flatMap(([heading,body,uri])=>[
    {bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:180,runs:[{run:heading,style:{bold:true}}]},
    {bulletCharacter:'',marginLeft:0,indent:0,spaceBefore:0,spaceAfter:size==='17pt'?700:900,runs:[{run:body,...(uri?{link:{uri,isExternal:true},style:{color:'#000000',underline:'sng'}}:{})}]}
  ]);
  shape.text.style={typeface:'Calibri',fontSize:size,color:'#000000',alignment:'left',verticalAlignment:'top',autoFit:'none',lineSpacing:1.04,insets:{top:0,bottom:0,left:9.6,right:9.6}};
  prose.push(...items.flatMap(x=>[x[0],x[1]]));
}

setTitle(original[0],'Inference optimization through\nmulti-token prediction and\nspeculative decoding','42pt');
original[0].shapes.items[1].text='Ayo Adetayo (aa5886)\nAnh Nguyen (ahn2130)\nPilar Maldonado (pjm2109)\nQuang Nguyen (qnn2102)';
original[0].shapes.items[1].text.style={typeface:'Calibri',fontSize:'24pt',color:'#000000',alignment:'center',autoFit:'none'};
original[0].speakerNotes.text='Team 34. Source: PROJECT/proposal/inference-optimization-proposal.tex. This presentation summarizes the completed baseline and pilot evaluation from November 22–23, 2026.';

setTitle(original[1],'Project summary');
sections(original[1],[
 ['Project goal','The project tests whether smaller draft networks accelerate generation while preserving the target model’s responses.'],
 ['Background context','Speculative decoding verifies several draft tokens together. Acceptance and execution time determine its speed.'],
 ['Optimization and technical methodology','The study compares EAGLE-3, P-EAGLE, DFlash, and a pilot with 100%, 75%, and 50% feed-forward widths.'],
 ['Profiling, benchmarking, and evaluation tools','The client measures throughput and latency. Native counters, Slurm, and GPU samples record acceptance and compute.'],
 ['Experiment tracking dashboard','The Weights & Biases trial dashboard contains measurements and run records.','https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22'],
 ['GitHub repository','The project repository contains the evaluation code and pilot implementation.','https://github.com/einnuian/hpml-project']
],'19pt');
original[1].speakerNotes.text=reportSource+'\nThe pilot uses the team’s custom four-layer P-EAGLE-style drafter and imported checkpoints. Its results remain separate from the released P-EAGLE reproduction. OTPS means output tokens per second. GPU time is compute usage. The project reports no monetary charge.\nDashboard: https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22\nRepository: https://github.com/einnuian/hpml-project';

setTitle(original[2],'Current progress');
sections(original[2],[
 ['Baseline and experimental setup','Nine configurations complete 810 timed responses using Llama-3.1-8B-Instruct and NVIDIA L40S GPUs. Each configuration uses the same 30 prompts, three repetitions, one active request, greedy decoding, and a 2,048-token output limit.'],
 ['Profiling and baseline analysis','DFlash leads six benchmarks in throughput and GPU time per token. P-EAGLE leads MATH-500 and AIME 2025. EAGLE-3 leads MT-Bench and CNN/DailyMail. Exact response differences remain unresolved.'],
 ['First optimization and implementation status','All three Pilot FFN widths complete evaluation. The 75% and 50% variants are slower than their matched ordinary generation (AR) reference on every benchmark. The 75% width exceeds 50% throughput on seven benchmarks. The 100% run uses another physical GPU.']
],'20pt');
original[2].speakerNotes.text=reportSource+'\nThe ten benchmarks are MT-Bench, Alpaca, Arena-Hard, CNN/DailyMail, GSM8K, MATH-500, AIME 2025, HumanEval, MBPP, and LiveCodeBench. The first three selected prompts from each benchmark form the 30-prompt set. DFlash leads six initial benchmark OTPS means. P-EAGLE leads MATH-500 and AIME 2025. EAGLE-3 leads MT-Bench and CNN/DailyMail. The 75% speedup range is 0.798–0.971, and the 50% range is 0.807–0.936. Both narrowed pilots match their AR reference on 12 of 30 distinct prompts. These are observed execution results, and benchmark task accuracy remains unmeasured. The complete original 100% attempt uses another physical L40S and has an incomplete matching AR reference.';

const results=[
 {title:'Mathematics throughput results',category:'mathematics',metric:'OTPS',page:4,caption:'P-EAGLE leads MATH-500 and AIME 2025, and DFlash leads GSM8K in this mathematics excerpt.'},
 {title:'Mathematics GPU compute usage',category:'mathematics',metric:'GPU time',page:6,caption:'Lower GPU time per output token indicates greater efficiency, with P-EAGLE leading two mathematics benchmarks and DFlash leading GSM8K.'},
 {title:'Code throughput results',category:'code',metric:'OTPS',page:4,caption:'DFlash has the highest observed throughput on all three code benchmarks. Both narrowed pilots remain below their matched AR reference.'},
 {title:'Code GPU compute usage',category:'code',metric:'GPU time',page:6,caption:'DFlash uses the least GPU time per output token on all three code benchmarks. Both narrowed pilots use more GPU time per token than their matched AR reference.'},
 {title:'Conversation and summarization\nthroughput results',category:'conversation, instruction following, and summarization',metric:'OTPS',page:4,caption:'EAGLE-3 leads MT-Bench and the CNN/DailyMail summarization benchmark. DFlash leads the Alpaca and Arena-Hard instruction-following benchmarks.'},
 {title:'Conversation and summarization\nGPU compute usage',category:'conversation, instruction following, and summarization',metric:'GPU time',page:6,caption:'EAGLE-3 uses the least GPU time per output token on MT-Bench and CNN/DailyMail. DFlash leads Alpaca and Arena-Hard on the same metric.'}
];
const resultSlides=[];
let previousSlide=original[2];
for(let i=0;i<results.length;i++) {
 previousSlide=previousSlide.duplicate();
 resultSlides.push(previousSlide);
}
for(let i=0;i<results.length;i++) {
 const s=resultSlides[i];
 const result=results[i];
 setTitle(s,result.title);
 const unit=s.shapes.add({geometry:'textbox',position:{left:97.6,top:result.title.includes('\n')?183:151,width:1084.8,height:30},fill:'none',line:{fill:'none',width:0}});
 unit.text=result.metric==='OTPS'?'Output tokens per second (tokens/s)':'Allocated GPU time (GPU-seconds per million output tokens)';
 unit.text.style={fontSize:'17pt',typeface:'Calibri',color:'#000000',autoFit:'none',insets:0};
 const body=s.shapes.items[1];
 body.text='';
 const imagePath=screenshotPaths[i];
 const bytes=await fs.readFile(imagePath);
 s.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:'The project report’s '+result.category+' panel for '+result.metric+'.',fit:'contain',position:{left:88,top:result.title.includes('\n')?220:184,width:1104,height:result.title.includes('\n')?365:405}});
 const caption=s.shapes.add({geometry:'textbox',position:{left:97.6,top:604,width:1084.8,height:74},fill:'none',line:{fill:'none',width:0}});
 caption.text=result.caption;
 caption.text.style={fontSize:'18pt',typeface:'Calibri',color:'#000000',autoFit:'none',insets:0};
 prose.push(caption.text.toString());
 s.speakerNotes.text=reportSource+'\nThe '+result.category+' results cover the same prepared prompts and three repetitions. '+(result.metric==='OTPS'?'OTPS is the delivered output-token count divided by the complete three-prompt block interval. Higher OTPS indicates faster output delivery.':'Timed allocated GPU time equals the benchmark interval multiplied by one GPU. The normalized result divides that time by delivered tokens. The table unit is GPU-seconds per million output tokens. This metric measures compute usage. It includes client gaps within the block. Model loading and warmups have separate allocation accounting.')+'\nEntries show the mean, sample standard deviation, and repetition count in brackets. All statements describe observed execution means. Response differences and the small prompt sample remain limitations.\nOverleaf source: https://www.overleaf.com/project/6ac8113d9bf3877de5dcecf2 , PDF page '+result.page+'.\nScreenshot source: '+imagePath;
}

setTitle(original[3],'Future work');
sections(original[3],[
 ['Remaining work and experiments','The next experiments strengthen the full-width drafter and test conditioning, feed-forward compression, and alternative drafting architectures. The goal is higher throughput and lower GPU time per token than ordinary generation and the existing methods. A controlled width comparison will use one physical GPU and fixed training conditions.'],
 ['Scope adjustments','The evaluation will expand to Qwen3.8-27B and Inkling 2 Mini after verifying runtime support and compatible drafters. Larger prompt samples and task-specific scoring will test whether performance gains generalize. Exact response checks will accompany each new configuration.']
],'20pt');
original[3].speakerNotes.text=reportSource+'\nFuture work describes planned research. No additional model evaluation has run. Qwen3.8-27B became available August 14, 2026; the official FP8 checkpoint is a candidate for L40S compatibility checks. Sources: https://github.com/QwenLM/Qwen3.8#news and https://huggingface.co/Qwen/Qwen3.8-27B-FP8 . Inkling 2 Mini launched November 20, 2026; its looped MoE architecture and mixed NVFP4/MXFP8 checkpoint require separate engine and drafter compatibility checks. Sources: https://thinkingmachines.ai/news/inkling-2-mini/ and https://huggingface.co/thinkingmachines/Inkling-2-Mini . The checkpoints evaluated in the present pilot have unresolved original seed and model-revision provenance. A future controlled study records those conditions explicitly.';

setTitle(original[4],'Blockers and limitations');
sections(original[4],[
 ['Blockers and system limitations','Limited GPU availability in Colab interrupted the initial workflow. The project moved to Columbia’s Insomnia cluster, where queue times and allocation limits still affect experiment scheduling. Measurements across different physical GPUs also limit direct width comparisons.'],
 ['Resolved issues','Authenticated cluster access, isolated software environments, and a repeatable streaming evaluator now support the complete trial. Saved prompts, checkpoint hashes, output tokens, measurements, and tracking records make the completed evaluation reproducible.'],
 ['Remaining limitations','The next study will repeat all widths on one physical GPU and investigate the response differences. Three prompts per benchmark provide an initial sample. Incomplete memory coverage and unavailable pilot component timing limit the current resource analysis.']
],'20pt');
original[4].speakerNotes.text=reportSource+'\nThe continuation records complete 75%, 50%, and AR results. The intended 100% replay fails during launch preparation before model inference begins. The report therefore retains the completed original 100% run on a different physical L40S. The continuation memory sampler begins during AR, after the two narrowed pilots complete. Their device-memory values remain empty. Pilot framework-memory peaks and phase timings remain unavailable. Original team checkpoints contain 200 training examples and 50 optimizer updates per width; the original training seed and target revision are unresolved.';

await fs.mkdir(outputDir,{recursive:true});
await fs.writeFile(path.join(outputDir,'authored-prose.txt'),prose.join('\n\n'));
for(let i=0;i<deck.slides.items.length;i++) {
  const slide=deck.slides.items[i];
  await fs.writeFile(path.join(outputDir,`slide-${i+1}.png`),new Uint8Array(await (await deck.export({slide,format:'png',scale:1})).arrayBuffer()));
}
await (await PresentationFile.exportPptx(deck)).save(path.join(outputDir,'candidate.pptx'));
await fs.writeFile(path.join(outputDir,'final-inspect.ndjson'),(await deck.inspect({select:'$.slides[*]',include:['id','index','layout','elements.id','elements.text','elements.position','elements.style','notes.text']})).ndjson);
console.log('Prepared',deck.slides.items.length,'slides at',outputDir);
