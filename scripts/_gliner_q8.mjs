const B = "./browser.mjs";
const { closePage, launchChrome, openPage, startServer, CDP } = await import(B);
const { port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const evalL = (sid, expr, ms = 300000) =>
  cdp.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }, sid, ms).then((r) => r.result?.value);
try {
  const pg = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/models/`);
  const r = await evalL(pg.sessionId, `(async()=>{
   try {
    const T = await import("/web-ai-showcase/lib/webai.js").then(m=>import(m.TRANSFORMERS_URL));
    const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs");
    ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/"; ort.env.wasm.numThreads=1;
    const REPO="onnx-community/gliner_small-v2.1", MAXW=12;
    const tok = await T.AutoTokenizer.from_pretrained(REPO);
    const sess = await ort.InferenceSession.create(new Uint8Array(await (await fetch("https://huggingface.co/"+REPO+"/resolve/main/onnx/model_quantized.onnx")).arrayBuffer()),{executionProviders:["wasm"]});
    const SPLIT=/\\w+(?:[-_]\\w+)*|\\S/g; const sigmoid=(x)=>1/(1+Math.exp(-x));
    const hasOv=(a,b)=>{ if(a.slice(0,2).toString()===b.slice(0,2).toString())return true; if(a[0]>b[1]||b[0]>a[1])return false; return true; };
    async function extract(text, entities, threshold=0.5){
      const words=[],ws=[],we=[]; let m; SPLIT.lastIndex=0;
      while((m=SPLIT.exec(text))!==null){words.push(m[0]);ws.push(m.index);we.push(SPLIT.lastIndex);}
      const textLength=words.length;
      const promptItems=[]; for(const e of entities){promptItems.push("<<ENT>>");promptItems.push(e);} promptItems.push("<<SEP>>");
      const promptLength=promptItems.length; const seq=promptItems.concat(words);
      let wordsMask=[0],inputIds=[1],attn=[1],c=1;
      seq.forEach((word,wordId)=>{ const sub=tok.encode(word).slice(1,-1); sub.forEach((t,ti)=>{ attn.push(1); if(wordId<promptLength)wordsMask.push(0); else if(ti===0){wordsMask.push(c);c++;} else wordsMask.push(0); inputIds.push(t); }); });
      wordsMask.push(0); inputIds.push(tok.sep_token_id); attn.push(1); const L=inputIds.length;
      const spanIdx=[],spanMask=[];
      for(let i=0;i<textLength;i++)for(let j=0;j<MAXW;j++){const e=Math.min(i+j,textLength-1);spanIdx.push(i,e);spanMask.push(i+j<textLength?1:0);}
      const numSpans=textLength*MAXW; const bi=(a)=>BigInt64Array.from(a.map(BigInt));
      const out=await sess.run({input_ids:new ort.Tensor("int64",bi(inputIds),[1,L]),attention_mask:new ort.Tensor("int64",bi(attn),[1,L]),words_mask:new ort.Tensor("int64",bi(wordsMask),[1,L]),text_lengths:new ort.Tensor("int64",bi([textLength]),[1,1]),span_idx:new ort.Tensor("int64",bi(spanIdx),[1,numSpans,2]),span_mask:new ort.Tensor("bool",Uint8Array.from(spanMask),[1,numSpans])});
      const logits=out.logits.data; const numEntities=entities.length; const idToClass={}; entities.forEach((e,i)=>idToClass[i+1]=e);
      const startTokPad=MAXW*numEntities,endTokPad=numEntities; const spans=[];
      logits.forEach((value,id)=>{ const st=Math.floor(id/startTokPad)%textLength; const et=st+Math.floor(id/endTokPad)%MAXW; const en=id%numEntities; const p=sigmoid(value); if(p>=threshold&&st<textLength&&et<textLength)spans.push([text.slice(ws[st],we[et]),ws[st],we[et],idToClass[en+1],p]); });
      const sorted=spans.slice().sort((a,b)=>b[4]-a[4]); const keep=[];
      for(const s of sorted){ let ov=false; for(const k of keep)if(hasOv([s[1],s[2]],[k[1],k[2]])){ov=true;break;} if(!ov)keep.push(s); }
      return keep.sort((a,b)=>a[1]-b[1]).map(s=>s[3]+": "+s[0]+" ("+s[4].toFixed(2)+")");
    }
    const t0=performance.now();
    const ex1=await extract("Barack Obama was born in Honolulu Hawaii",["person","location"]);
    const ms=Math.round(performance.now()-t0);
    return {ok:true, ms, ex1, ex2: await extract("Apple was founded by Steve Jobs in California in 1976",["company","person","location","date"]), ex3: await extract("Tesla was recalled after a bug the CEO Elon Musk apologized on Monday",["company","person","product","day"])};
   } catch(e){ return {ok:false, err:String(e&&e.stack||e&&e.message||e).slice(0,300)}; }
  })()`, 300000);
  console.log("RESULT:"+JSON.stringify(r,null,1));
  await closePage(cdp, pg.targetId);
} finally { chrome.kill(); process.exit(0); }
