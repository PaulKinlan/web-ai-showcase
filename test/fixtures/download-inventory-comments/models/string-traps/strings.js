// Strings and regexes below contain comment OPENERS as data. Masking must not treat them as comments
// (that would corrupt the executable text) and must still mask the real trailing comment.
const url = "https://huggingface.co/fixture//model/*shard*/";
const pattern = /\/\*\*/;
const template = `a // b /* c */ d`;
import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

export const score = () => loadPipeline({ task: "image-classification", model: "fixture/model" }); // ort.env named in a real comment
