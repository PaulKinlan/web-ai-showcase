// Executable ORT usage — the tokens below are CODE, so this route must stay raw-ort.
import * as ort from "onnxruntime-web";

export const open = () => ort.InferenceSession.create("/fixture/model.onnx");
