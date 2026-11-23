# Expert study protocol (RQ4)

**RQ4.** What do LLM-evaluation experts discover with the tool, and how do they use the consensus
and leaderboard views to judge whether a flagged question is broken and whether it matters?

This is a formative, qualitative study, following the proposal feedback. It has no control
condition, no timing, no comparison between tools and no statistical claims. We report what
participants did and said, as themes with quotes.

## At a glance

| | |
|---|---|
| Participants | 3 to 5 people who work on LLM evaluation, NLP benchmarks, or data quality |
| Length | 45 minutes, one participant per session |
| Format | Think-aloud over a video call or in person, with screen sharing |
| People | Facilitator (runs the script) and note-taker (fills the log). Both of us, swapping roles. |
| Tool | Study mode: `http://localhost:8766/?study`, or the hosted link with `?study` added |
| Output | One filled log per session (`study/sessions/P1.md`, …), then the theme table in the report |

**Study mode** hides every MMLU-Redux label: the colors, legend, presets, badges, corrections,
notes and the detection table. Participants judge questions from the model evidence alone, and we
compare their verdicts with Redux afterwards. It also adds a **Study set (8)** button that selects
the fixed triage questions.

## Before the sessions

- [ ] **Pilot:** Arman and Padmapriya each run the full script on the other once. Fix confusing wording before P1.
- [ ] **Recruit:** ask NYU CDS PhD students, people from the course who did evaluation projects, or lab members working on LLM evaluation. Aim for 4 and accept 3. Record only role and experience (for example "PhD student, 2 years on LLM evaluation"), never names.
- [ ] **Check the build:** the study set loads 8 questions, nothing on screen says "Redux", the leaderboard shows a p-value.
- [ ] **Recording:** make audio or screen recordings only with consent, and only for note-taking; delete them after coding. Without consent, the note-taker's log is the record.
- [ ] Copy `session_log_template.md` (at the end of this file) to `study/sessions/P<n>.md`.

## Session script

Read the quoted text close to verbatim, so every session starts the same. Text in *italics* is
for the facilitator.

### 0. Welcome and consent (3 min)

> "Thanks for helping. This is a course project for DS-GA 3001. We built a tool for finding
> broken questions in the MMLU benchmark and want to see how someone with evaluation experience
> uses it. We're testing the tool, not you; there are no right answers, and getting confused is
> useful to us. Please think out loud the whole time: say what you're looking at, what you
> expect, and what surprises you. We'll take notes but won't record your name. [If recording:]
> Is it okay if we record audio, only for our notes, deleted afterwards? You can stop at any time."

*Record the participant's role, years of experience, and how familiar they are with MMLU and with IRT on the log.*

### 1. Guided tour (7 min)

*Share the tool in study mode. Walk through it once with the same wording, and don't interpret the results.*

> "Each dot is one of the 14,042 MMLU questions. We ran 150 public models on all of them, using
> results from the archived Open LLM Leaderboard. The x-axis is how much probability the models
> put on the official answer key on average. The y-axis is how much the models disagree about
> it. The color is how strongly the 30 strongest models agree on one alternative answer."
>
> "Clicking a point opens the inspector: the question, the official key, the 'consensus
> alternative' (the answer the strongest models prefer apart from the key), how all models and
> the top 30 voted, and a plot of each model's probability on the key against its overall
> accuracy."
>
> "At the bottom, the leaderboard view recomputes the ranking of the 150 models if the selected
> questions are removed or re-keyed to the consensus. It compares the result with removing random
> questions of the same difficulty, which gives a p-value."
>
> "The Models menu recomputes everything with a subset of the models, such as base models only
> or a random half."

### 2. Free exploration (7 min)

> "Take a few minutes to explore. Where on this map would you look first for broken questions,
> and why?"

*Probe only if they go quiet: "What are you looking at now?" or "What did you expect there?" Note which axes and colors they choose, and where they brush.*

### 3. Triage the study set (15 min), the core task

*Click **Study set (8)**. Go through the questions in the table order, or the order they choose.*

> "Here are 8 questions the tool considers suspicious. For each one, please tell us which of
> these it is:
> **(a)** the answer key is wrong, **(b)** the question is ambiguous or has more than one
> defensible answer, **(c)** it's a hard question and the key is right, or **(d)** you can't tell.
> Say how confident you are, from 1 to 5, and what on the screen convinced you."

*For each question, log the verdict, the confidence, and the evidence they cite (their own knowledge, vote bars, top-tier vs. all models, the P(key)-vs-accuracy plot, scores, IRT). Don't confirm or correct any verdict. If they spend more than about 3 minutes on one question, say "Let's move on; you can come back to it."*

### 4. Does it matter? (8 min)

> "Now select the questions you judged wrong-key or ambiguous (Shift-drag on the map, or start
> from the study set). Look at the leaderboard view. Would you change how you report MMLU
> results because of these questions? What tells you that?"

Follow-ups, asked in this order, stopping when time runs out:
1. "Try **Re-key to consensus** instead of Remove. Does the answer change?"
2. "What do you make of the comparison with random removals and its p-value?"
3. "Switch Models to *Pretrained base models only*, or to one of the random halves. Does your conclusion hold?"
4. *(Optional, if time allows:)* "Use **Top-tier conf. < 0.2** to select about a thousand questions, and try Remove again."

### 5. Debrief (5 min)

1. "What was the most useful thing on the screen for deciding whether a question is broken?"
2. "What did you not trust, or find confusing?"
3. "Where would a tool like this fit in your own evaluation work, if anywhere?"
4. "If you could add one thing, what would it be?"
5. "Compared with a sorted spreadsheet of these scores, what if anything does the map add?" *(Opinion only. This is not a test against a spreadsheet.)*

> "Thank you, that's everything."

## Facilitator rules

- **Don't reveal the Redux labels or the answer key** (`facilitator_key.md`) until the session is over. Afterwards you may tell them how their verdicts compared, if they ask.
- **Don't teach.** If they misread a view, note it as a finding. Correct it only if they're stuck for more than a minute, and log that you intervened.
- Use neutral prompts: "What are you thinking?", "What would you do next?", "Say more about that." Avoid "Did you notice…".
- Keep to time. Sections 3 and 4 must happen in every session; trim section 2 first.

## Analysis plan

1. **Verdicts vs. reference.** For each of the 8 questions, put each participant's verdict next to the Redux label. For the three questions Redux did not review, use our own check in `facilitator_key.md`. Report this as a table of agreements and disagreements, not as an accuracy score. With 3 to 5 people it is descriptive only.
2. **Evidence used.** Count how often each view or score was cited as decisive: vote bars, top-tier vs. all, the P(key)-vs-accuracy plot, consensus alternative, IRT, leaderboard. This speaks to the grader's point about what the map adds beyond IRT.
3. **Themes.** Both of us code the logs on our own, then merge into one code book. Starting codes: *evidence used*, *trust and doubt*, *misreadings*, *judging consequence*, *robustness reasoning*, *feature requests*, *usability problems*. Add new codes as they appear, and agree on them together.
4. **Report.** In the RQ4 section: a participant table (role and experience only), the verdict table, 3 to 5 themes, each with 1 or 2 short quotes, and the usability problems we fixed or left. State the limits: few participants, and colleagues rather than a sample of practitioners.

---

## Session log template

Copy this into `study/sessions/P<n>.md`, one file per participant.

```markdown
# Session P<n>

- Date:            Facilitator:            Note-taker:
- Role / experience:
- Familiarity: MMLU (none / some / a lot)   IRT (none / some / a lot)
- Recording consent: yes / no

## 2. Free exploration
- Where they looked first, and why:
- Axis, color, and model-set changes:
- Notable quotes:

## 3. Triage
| # | Question | Verdict (a/b/c/d) | Conf. 1-5 | Evidence cited | Notes / quotes |
|---|---|---|---|---|---|
| 1 | miscellaneous/136 | | | | |
| 2 | virology/93 | | | | |
| 3 | high_school_physics/83 | | | | |
| 4 | machine_learning/27 | | | | |
| 5 | miscellaneous/468 | | | | |
| 6 | high_school_mathematics/136 | | | | |
| 7 | abstract_algebra/59 | | | | |
| 8 | high_school_psychology/53 | | | | |

## 4. Does it matter?
- What they selected:
- Remove vs. re-key reading:
- Reading of the p-value / random baseline (correct? misread?):
- Model-subset check and conclusion:
- Would they change how they report MMLU? Why?

## 5. Debrief
1. Most useful:
2. Not trusted / confusing:
3. Fit in their work:
4. One addition:
5. Map vs. spreadsheet:

## Interventions and misreadings
-

## Note-taker's top 3 observations
1.
2.
3.
```
