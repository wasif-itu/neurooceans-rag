# Chat Seed Data Generation — Prompt & Schema

Use this to have another LLM generate WhatsApp/iMessage conversation content
for seeding LifeWeaver's chat-based story generation. Fill in the
placeholders in the **Prompt** section below, then paste the whole prompt
into the other LLM. Its output feeds directly into
`scripts/seeding_scripts/seed_diana_life_story_chat_data.py`-style seed
scripts (see that file for the exact pattern).

---

## Prompt (copy everything below the line, fill in placeholders)

```
You are generating realistic WhatsApp and iMessage conversation history
between two people, for seeding a test/demo database. Output must be a
single JSON object matching the schema below — nothing else, no markdown
fences, no commentary.

## Context

- Account owner ("me"): {{OWNER_NAME}} — {{OWNER_BRIEF_DESCRIPTION}}
- Contact: {{CONTACT_NAME}}
- Relationship: {{RELATIONSHIP}} (e.g. childhood best friend, sister, spouse)
- Personality of {{CONTACT_NAME}}: {{PERSONALITY_DESCRIPTION}}
  (e.g. "Warm, sarcastic, always lowercase, uses 'lol' and 'ngl' a lot,
  checks in constantly, blunt but affectionate, texts in bursts of 2-3
  short messages instead of one long one.")
- Time period / setting: {{TIME_PERIOD_OR_THEME}}
  (e.g. "1980-1997, following major life events" or "the last 6 months,
  casual day-to-day life")
- Story throughline: {{STORY_THROUGHLINE}}
  (e.g. "engagement, wedding, career milestones, a big move, a falling out
  and reconciliation — whatever gives the conversation a real arc")
- Approximate volume: {{MESSAGE_COUNT}} total messages, split across both
  "whatsapp" and "imessage" (e.g. "~60 messages, mostly whatsapp, a smaller
  imessage thread covering a different stretch of time")

## Writing style rules

- Casual, real texting voice: lowercase is fine, occasional typos, emoji,
  contractions, sentence fragments. Avoid formal or narrated prose — this
  must read like an actual text thread, not a story summary.
- Give {{CONTACT_NAME}} a consistent voice matching the personality above;
  give "me" a consistent, distinct voice too.
- Concrete and specific: name real events, dates, feelings, callbacks to
  earlier messages. Avoid generic filler ("that's great!", "how are you")
  as filler-only content — every exchange should carry some real detail,
  since this content is later retrieved and quoted by a story-generation
  model.
- Chronological order, spread across realistic dates and times of day (not
  everything on one day). Leave natural gaps (hours, days, sometimes weeks)
  between exchanges as the story moves forward.
- `is_from_me` alternates naturally as a real back-and-forth conversation
  would, not in a strict 1-for-1 pattern.
- Do not include real phone numbers, addresses, or other real-world PII.

## Output schema

Return exactly this shape:

{
  "whatsapp": [
    {"date": "YYYY-MM-DD", "hour": 0-23, "minute": 0-59, "is_from_me": true|false, "content": "message text"}
  ],
  "imessage": [
    {"date": "YYYY-MM-DD", "hour": 0-23, "minute": 0-59, "is_from_me": true|false, "content": "message text"}
  ]
}

Either array may be empty if the volume/theme calls for it, but both keys
must be present. Every object must have exactly these five fields, no more,
no fewer. `content` must be plain text (no markdown formatting, no HTML).
```

---

## Field reference (for whoever wires the output into the DB)

| Field | Type | Notes |
|---|---|---|
| `date` | `"YYYY-MM-DD"` string | Any date, past or present |
| `hour` | int, 0-23 | |
| `minute` | int, 0-59 | |
| `is_from_me` | bool | `true` = account owner, `false` = the contact |
| `content` | string | The message text — the part story-generation actually retrieves |

**Derived by the seed script, not the LLM:** message `id` (sequential,
script-generated), `sender` (`"Me"` vs. the contact's name, mapped from
`is_from_me`), `chat_jid` / `chat_id`, combined UTC `timestamp`. Chat-level
fields (`name`, `chat_type`, `sync_complete=True`) are also set once by the
script, not per message.

## Example output snippet

```json
{
  "whatsapp": [
    {"date": "2024-03-02", "hour": 19, "minute": 42, "is_from_me": false, "content": "ok i need to tell you something and you need to sit down"},
    {"date": "2024-03-02", "hour": 19, "minute": 43, "is_from_me": true, "content": "im sitting. go"},
    {"date": "2024-03-02", "hour": 19, "minute": 45, "is_from_me": false, "content": "we got the house!! the one with the weird kitchen we joked about"}
  ],
  "imessage": []
}
```

## Group chats (optional extension)

The schema above assumes exactly one contact ("me" vs. one other person).
For a group chat with multiple named participants, add a `"sender"` field
per message (e.g. `"sender": "Priya"`) instead of relying on `is_from_me`
alone, and note that in the prompt's Context section. The seed script needs
a small change to read `sender` directly rather than deriving it — flag
this if you need it and the script will be adjusted alongside the format.
