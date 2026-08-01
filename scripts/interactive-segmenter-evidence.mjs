export function screenshotBindingMatches({ shot, bytes, sha256, width, height, viewport }) {
  return Boolean(
    shot && bytes && viewport && shot.sha256 === sha256 && shot.bytes === bytes.length &&
      shot.image?.width === width && shot.image?.height === height && width === viewport.width &&
      height >= viewport.height && shot.expectedViewport?.width === viewport.width &&
      shot.expectedViewport?.height === viewport.height && shot.expectedViewport?.dpr === 1,
  );
}

export function sourceUsesExecutableMcp(captureSource, validatorSource) {
  return /new Client\(/.test(captureSource) && /StdioClientTransport/.test(captureSource) &&
    /client\.callTool/.test(captureSource) && /"click"/.test(captureSource) &&
    /"press_key"/.test(captureSource) && /"upload_file"/.test(captureSource) &&
    /"list_console_messages"/.test(captureSource) &&
    /"list_network_requests"/.test(captureSource) &&
    !/void\s*\[/.test(validatorSource);
}
