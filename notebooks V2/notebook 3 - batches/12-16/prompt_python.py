PROMPT_TEMPLATE = '''[INST]
Analyze this semantic job community and return ONLY valid JSON.

=== RESEARCH CONTEXT ===
You are classifying labor market communities to answer:
"What signals distinguish the ENTRY POINT into the labor market vs mid vs senior roles?"

=== ENTRY ACCESSIBILITY DEFINITIONS (CHOOSE EXACTLY ONE) ===

clear_entry_point:
- PRIMARY SIGNAL: Junior roles dominate (>60% junior_share)
- NO formal credentials required: requires_bachelor=0 AND requires_certification=0 in most postings
- Language patterns: "training provided", "no experience necessary", "entry-level", "will train", "learn on the job"
- Skill patterns: Generic/transferable skills (communication, teamwork, basic software), NOT specialized tools
- Role scope: Task execution, supervised work, narrow responsibility
- Examples: Retail cashier, food service, warehouse picker, general labor, data entry clerk
- Decision rule: IF junior_share > 0.6 AND requires_bachelor = 0 → clear_entry_point

moderate_entry_barrier:
- PRIMARY SIGNAL: Mixed experience levels OR junior roles with credential preferences
- Credentials PREFERRED but not mandatory: "Bachelor's preferred" OR "2-4 years experience preferred"
- Language patterns: "preferred", "plus", "nice to have", "or equivalent experience", "associate degree acceptable"
- Skill patterns: Domain-specific tools/frameworks (Excel, Salesforce, Python basics), some specialization
- Role scope: Independent contribution, some autonomy, moderate complexity
- Examples: Marketing coordinator, junior developer, IT support, admin assistant, nurse assistant
- Decision rule: IF (junior_share 0.4-0.6) OR (requires_bachelor = 1 AND junior_share > 0.5) → moderate_entry_barrier

restricted_entry_point:
- PRIMARY SIGNAL: Senior roles dominate OR specialized expertise mandatory
- Credentials REQUIRED: Advanced degree, professional license, OR 5+ years experience explicitly required
- Language patterns: "required", "must have", "license", "certification", "RN", "CPA", "PE", "CISSP", "PMP"
- Skill patterns: Advanced/specialized expertise (architectural design, clinical procedures, legal strategy)
- Role scope: Strategic leadership, decision-making authority, mentoring others, high-stakes outcomes
- Examples: Registered nurse (RN license), licensed attorney, senior engineer, medical doctor, CPA
- Decision rule: IF senior_share > 0.5 OR requires_masters = 1 OR requires_certification = 1 → restricted_entry_point

=== MID vs SENIOR DISTINCTION SIGNALS (for dominant_experience_level) ===

mid-level indicators:
- Language: "implement", "maintain", "optimize", "contribute", "coordinate", "hands-on"
- Responsibility: Execute defined projects, collaborate with team, report to manager
- Skills: Technical proficiency in domain tools, growing specialization
- Autonomy: Work independently on assigned tasks, seek guidance for complex issues

senior-level indicators:
- Language: "lead", "architect", "strategy", "stakeholder", "oversee", "ownership", "drive", "mentor"
- Responsibility: Define direction, make strategic decisions, mentor others, own outcomes
- Skills: Deep expertise, cross-functional knowledge, ability to translate business needs to technical solutions
- Autonomy: Set priorities, influence organizational decisions, represent team to leadership

=== CLASSIFICATION PRIORITY ORDER ===
1. First determine dominant_experience_level using language patterns + share distribution
2. Then determine entry_accessibility using credential requirements + junior_share + language signals
3. If signals conflict, prioritize CREDENTIAL REQUIREMENTS over title keywords
   (e.g., "Junior Engineer" with "PE license required" → restricted_entry_point)

=== COMMUNITY DATA ===
community_id: {community_id}
community_size: {community_size}
top_keywords: {top_title_keywords}
junior_share: {junior_share}
mid_share: {mid_share}
senior_share: {senior_share}

signals:
{signals_text}

representative_rank: {rep_rank}
representative_text:
{rep_text}

=== OUTPUT REQUIREMENTS ===
Return JSON with EXACTLY these fields:
- community_label: Short descriptive name (3-6 words)
- dominant_experience_level: "junior" OR "mid" OR "senior"
- entry_accessibility: "clear_entry_point" OR "moderate_entry_barrier" OR "restricted_entry_point"
- defining_skills: List of 3-5 specific skills/tools mentioned in postings
- defining_responsibility_signals: List of 2-3 key responsibility patterns
- confidence: "low" OR "medium" OR "high" (based on clarity of signals in representative_text)

Return ONLY valid JSON. No explanations, no markdown, no extra text.
[/INST]'''