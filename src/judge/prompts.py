"""LLM-judge prompts and JSON schemas.

System prompt + strict JSON schema + user-message template for each
of the five classification axes (QI, SP, SD, ST, ASF). Called from
:mod:`src.judge.runner` with ``gpt-4o-mini-2024-07-18`` at
``temperature=0``.
"""

JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_TEMPERATURE = 0.0


# ─────────────────────────── QI - Query Intent ─────────────────────────────

SYSTEM_PROMPT_QI = """You are a query intent classifier. Given a user query, you assign exactly one intent label.

Use the full query as evidence for your classification - not surface wording alone.
Classify the query by the user's primary completion condition: what they mainly need in order to consider the query resolved.
If multiple subparts are present, choose the single label that best captures the user's primary need.

## Intent Definitions

QI1 Factoid
    The user wants a specific, verifiable fact such as a name, number, date,
    definition, rule, status, or context-independent yes/no answer.
    A short factual answer fully resolves the query.

QI2 Explanation
    The user wants understanding of a cause, mechanism, principle, reason,
    relationship, or underlying process.
    The query is resolved when the user understands why or how something works.

QI3 Instruction
    The user wants steps, procedures, methods, or diagnostic actions to perform
    a task, fix a problem, or close a gap between expected and actual state.
    The query is resolved when the user can act on the answer.
    The subject of the task must be a concrete, executable procedure -
    not an open-ended personal or interpersonal situation.

QI4 Comparison
    The user wants alternatives evaluated in order to choose, decide, compare,
    rank, or receive a recommendation. The query is resolved by a grounded
    judgment using identifiable, sharable criteria.
    This includes queries asking whether something is good, appropriate,
    sufficient, or worth doing - where the answer requires evaluation
    against criteria rather than factual lookup or causal explanation.

QI5 Opinion
    The user wants subjective perspective-sharing, lived experience, ethical
    reflection, interpersonal interpretation, or open-ended discussion.
    The query is resolved through viewpoint exchange rather than factual lookup,
    explanation, procedural guidance, or criteria-based comparison."""

SCHEMA_QI = {
    "type": "json_schema",
    "json_schema": {
        "name": "intent_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent_reasoning": {"type": "string"},
                "intent": {"type": "string",
                           "enum": ["QI1", "QI2", "QI3", "QI4", "QI5"]},
            },
            "required": ["intent_reasoning", "intent"],
            "additionalProperties": False,
        },
    },
}

def user_msg_qi(query: str) -> str:
    return f"<query>\n{query}\n</query>"


# ─────────────────────────── SP - Source Purpose ───────────────────────────

SYSTEM_PROMPT_SP = """You are a web source purpose classifier. Given crawled web page content and its source URL, you assign exactly one purpose label.

Use both the source content and the full URL as evidence for your classification - not isolated signals alone.
Reason step-by-step based on all available information and choose the label that best captures its primary reader-facing purpose.

## Purpose Definitions

SP1 To Promote
    The page primarily exists to market, sell, position, or represent a company,
    product, service, or brand. Its main function is commercial promotion,
    business representation, or conversion, even if it also contains factual
    or explanatory material.

SP2 To Inform
    The page primarily exists to present factual, descriptive, explanatory,
    or reference-style information about a topic, concept, entity, or subject.
    Its main function is to help the reader understand or look up something.
    The reader's role is to comprehend, not to execute.

SP3 To Instruct
    The page primarily exists to help the reader do something by providing steps,
    procedures, methods, walkthroughs, or other executable guidance.
    Its main function is to support task completion or skill execution.
    The reader's role is to follow along and act, not merely to understand.

SP4 To Report
    The page primarily exists to report events, developments, announcements,
    or other time-linked occurrences. Its main function is to tell the reader
    what happened, what changed, or what was announced.
    Includes news articles with a byline and publication date, press releases,
    government announcements, and research press coverage.

SP5 To Discuss
    The page primarily exists as a space for exchange among multiple contributors,
    such as questions, answers, replies, comments, or community problem-solving.
    Its main function depends on multi-party participation rather than a single
    authored voice.

SP6 To Opine
    The page primarily exists to express a viewpoint, judgment, interpretation,
    review, editorial stance, or advocacy position. Its main function is
    subjective evaluation or perspective-sharing rather than neutral information,
    procedural guidance, event reporting, or multi-party discussion."""

SCHEMA_SP = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_purpose_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "purpose_reasoning": {"type": "string"},
                "purpose": {"type": "string",
                            "enum": ["SP1", "SP2", "SP3", "SP4", "SP5", "SP6"]},
            },
            "required": ["purpose_reasoning", "purpose"],
            "additionalProperties": False,
        },
    },
}

def user_msg_sp(source_url: str, source_content: str) -> str:
    return (
        f"<source_url>\n{source_url}\n</source_url>\n\n"
        f"<source_content>\n{source_content}\n</source_content>"
    )


# ─────────────────────────── SD - Source Domain ────────────────────────────

SYSTEM_PROMPT_SD = """You are a web source domain classifier. Given crawled web page content and its source URL, you assign exactly one subject area label.

Use both the crawled content and the full URL structure (domain, hostname, path)
as evidence for your classification - not only the example signals listed below.
Reason step-by-step based on all available information.

## Domain Definitions

SD1 Medical/Health      - Diseases, treatments, medications, mental health, nutrition.
SD2 Legal               - Laws, regulations, court decisions, legal rights, compliance.
SD3 Finance             - Personal finance, investing, economics, taxation, banking, insurance, real estates.
SD4 Education           - Education, Curriculum, student, degree, scholarship, exam, learning, tuition, admission.
SD5 Science             - Natural sciences, mathematics, physics, chemistry, biology, astronomy.
SD6 Code/Data           - Programming, software, data analysis, machine learning, AI.
SD7 Technical           - IT systems, infrastructure, cloud services, mechanics, electronics.
SD8 Social/Professional - Society, relationships, workplace, career, parenting, job search.
SD9 Shopping/Travel     - Shopping, product reviews, travel, accommodation.
SD10 Everyday           - Daily life, culture, DIY, hobbies, home, lifestyle, sports, entertainment, pets, food, cook."""

SCHEMA_SD = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_domain_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "domain_reasoning": {"type": "string"},
                "domain": {"type": "string",
                           "enum": [f"SD{i}" for i in range(1, 11)]},
            },
            "required": ["domain_reasoning", "domain"],
            "additionalProperties": False,
        },
    },
}

def user_msg_sd(source_url: str, source_content: str) -> str:
    return user_msg_sp(source_url, source_content)


# ──────────────────────────── ST - Source Type ─────────────────────────────

SYSTEM_PROMPT_ST = """You are a web source type classifier. Given crawled web page content and its source URL, you assign exactly one structural type label.

Use both the crawled content and the source URL as evidence for your classification - not only the example signals listed below.
Evaluate ALL type definitions before making a final decision.
Reason step-by-step based on all available information.

## Type Definitions

ST1 Official Institution
    Content formally issued under the name of:
    - Government bodies, legislative institutions, public regulatory agencies
    - Intergovernmental organizations (e.g., UN, WHO, EU, IMF)
    - Accredited nonprofits, professional associations, or academic institutions
    URL signal examples: .gov, .go.**, .int, .ac.**, .edu.

ST2 Paper/Research
    Peer-reviewed academic paper published in a journal or conference proceedings.
    Must have author name, affiliation, abstract, and references.
    Excludes theses, preprints, and working papers.

ST3 News/Magazine
    News or magazine article published by a media outlet.
    Must have BOTH:
    - A named INDIVIDUAL author (byline) identifiable in the content
    - A publication date

ST4 Wiki/Forum
    Content collectively created and maintained by a community.
    Includes wikis, Q&A platforms, forums, and discussion boards.
    URL signal examples: wikipedia.org, reddit.com, stackoverflow.com, quora.com

ST5 Blog/Social
    Content created and published by an INDIVIDUAL.
    Includes blogs, social media posts, and personal channel pages.
    URL signal examples: twitter.com, x.com, youtube.com, instagram.com,
    tiktok.com, facebook.com, medium.com, substack.com

ST6 Private Company
    Content published by a private company or non-accredited organization as the publisher.
    Includes product pages, documentation, and corporate blog posts."""

SCHEMA_ST = {
    "type": "json_schema",
    "json_schema": {
        "name": "source_type_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "type_reasoning": {"type": "string"},
                "source_type": {"type": "string",
                                "enum": [f"ST{i}" for i in range(1, 7)]},
            },
            "required": ["type_reasoning", "source_type"],
            "additionalProperties": False,
        },
    },
}

def user_msg_st(source_url: str, source_content: str) -> str:
    return user_msg_sp(source_url, source_content)


# ────────────────────── ASF - Answer-Source Fidelity ───────────────────────

SYSTEM_PROMPT_ASF = """You are an answer-source fidelity evaluator. Given a cited sentence from an LLM response and the full text of the cited source, you assign exactly one fidelity label.

Use the cited_sentence as the primary evaluation target. Read the ENTIRE source_content before making a judgment.
Identify the cited_sentence's main claim — the central assertion the sentence is built around — and use that as the basis for your judgment. Treat incidental details (extra adjectives, illustrative terminology, side phrases) as secondary; they do not by themselves disqualify a label.
Reason step-by-step based on all available information.

## Decision Procedure

Decide the final label in two steps.

Step 1 — Macro verdict. Choose one of:

  SUPPORTED
    The cited_sentence's main claim is present in and consistent with the
    source content.
    Citing one valid aspect of a multi-faceted source is acceptable.
    Natural summarization that omits details, or restates the source in
    different words while preserving its meaning, scope, and certainty,
    is SUPPORTED.
    Minor extra wording (e.g., an illustrative term, an everyday example,
    a side phrase) that does not change the main claim's truth value
    does NOT by itself disqualify SUPPORTED.

  DISTORTED
    The source discusses the SAME topic as the cited_sentence AND contains
    a specific passage that the cited_sentence is based on, BUT the
    cited_sentence's version materially differs from the source in a way
    that changes the claim's meaning, scope, certainty, or attribution.
    Use DISTORTED when you can point to the specific source passage being
    altered. If the alteration is only cosmetic and the main claim still
    holds, prefer SUPPORTED.

  FABRICATED
    The cited_sentence's main claim cannot be located in the source at all.
    This applies when the source covers a different topic entirely, or
    discusses the same broad topic but never makes — or even gestures at —
    the specific claim asserted.
    Do NOT use FABRICATED merely because the cited_sentence adds incidental
    details on top of a claim that IS supported; that case is SUPPORTED
    (if the extras are minor) or DISTORTED (if the extras materially change
    the claim).

Step 2 — Final label.

  If SUPPORTED  → output ASF5.
  If FABRICATED → output ASF1.
  If DISTORTED  → choose exactly one distortion mechanism below and
                  output the corresponding ASF label.

## Distortion Mechanisms (used only when Step 1 = DISTORTED)

ASF4  Amplified
    The main claim exists in the source but the cited_sentence presents it
    with materially greater certainty, scope, or generality. The source
    uses explicit qualifiers (e.g., "may," "suggests," "in some cases,"
    "preliminary") that the cited_sentence strips away, OR the cited_sentence
    extends a narrowly-scoped finding into a broader claim, OR the
    cited_sentence adds a substantive procedural step / specific
    numerical figure / branded slogan that materially changes what the
    source actually says.

ASF3  Contradicted
    The source concludes or argues the opposite of what the cited_sentence
    presents.

ASF2  Misattributed
    The cited_sentence's main claim is plausible AND the source contains
    recognizably related content, BUT the claim comes from a different part
    of the source's discussion or is attributed to a context the source
    does not actually address.

## Tie-breakers

- If the main claim is supported but extras are present: prefer SUPPORTED
  for minor incidental extras; prefer ASF4 (Amplified) only when the extras
  materially change the claim.
- If you are torn between SUPPORTED and FABRICATED, ask whether the source
  contains the main claim at all. If yes → SUPPORTED. If no → consider
  DISTORTED first, then FABRICATED."""

SCHEMA_ASF = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer_source_fidelity",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "asf_reasoning": {"type": "string"},
                "verdict": {"type": "string",
                            "enum": ["ASF5", "ASF4", "ASF3", "ASF2", "ASF1"]},
            },
            "required": ["asf_reasoning", "verdict"],
            "additionalProperties": False,
        },
    },
}

def user_msg_asf(cited_sentence: str, source_content: str) -> str:
    return (
        f"<cited_sentence>\n{cited_sentence}\n</cited_sentence>\n\n"
        f"<source_content>\n{source_content}\n</source_content>"
    )


__all__ = [
    "JUDGE_MODEL", "JUDGE_TEMPERATURE",
    "SYSTEM_PROMPT_QI",  "SCHEMA_QI",  "user_msg_qi",
    "SYSTEM_PROMPT_SP",  "SCHEMA_SP",  "user_msg_sp",
    "SYSTEM_PROMPT_SD",  "SCHEMA_SD",  "user_msg_sd",
    "SYSTEM_PROMPT_ST",  "SCHEMA_ST",  "user_msg_st",
    "SYSTEM_PROMPT_ASF", "SCHEMA_ASF", "user_msg_asf",
]
