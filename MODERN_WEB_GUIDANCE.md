# modern-web-guidance — source, version, and update path

All frontend work in web-ai-showcase (HTML/CSS/client-side JS — new pages AND targeted fixes) MUST
consult modern-web-guidance FIRST for the specific UI/API topic and apply (or justify an exception
to) its recommendations. See the "modern-web-guidance is mandatory for all frontend work" policy in
`CLAUDE.md`, `AGENTS.md`, `.agents/skills/web-ai-showcase/SKILL.md`, and `.claude/routine-prompt.md`.

## Source (no vendored copy — never fork the guide text)

- **Canonical skill:** `modern-web-guidance` (the user/settings agent skill). Interactive agent runs
  invoke it directly.
- **Scripted fallback (routines / CI / non-interactive):** the published npm package, always resolved
  fresh from `@latest`:
  - Search: `npx -y modern-web-guidance@latest search "<specific task query>"`
  - Retrieve a guide by id: `npx -y modern-web-guidance@latest retrieve "<id>"`
- **Version / update path:** we deliberately DO NOT pin or vendor a copy — the guide evolves and web
  APIs change fast. Routines and contributors always call `@latest` so the guidance stays current.
  Pin only if a reproducibility incident ever requires it, and record the pin here with the reason.
- **Last confirmed pattern (informational, not a pin):** `modern-web-guidance@latest` as of
  2026-07-19. Refresh by re-running the commands above; there is nothing to bump in-repo.

## Recording consultation (enforced)

- Every frontend-touching critique (`models/<slug>/_questions.json`) records the guidance it consulted
  in `guidanceConsulted[]` (`{ id|query, recommendation, appliedOrException, evidence }`). An empty
  `guidanceConsulted` on a frontend critique is INCOMPLETE — `scripts/check-conformance.mjs` (via
  `validateCritique`) fails it.
- Each conformance suite carries a `build-process` assertion (`guidance-consulted`): "the frontend
  implementation consulted modern-web-guidance for its topics and applied/justified them."
- Topics that always warrant a lookup here: responsive control panels without horizontal overflow,
  accessible popover/dialog dismissal + focus trapping, streaming progress without INP regressions,
  loading/error/retry states, image/model loading + caching, modern CSS, and the browser APIs each
  demo uses (WebGPU, Web Workers, File/media APIs).
