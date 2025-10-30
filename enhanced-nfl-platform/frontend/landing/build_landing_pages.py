"""Generate marketing landing pages from a single template."""

from __future__ import annotations

from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT
OUTPUT_DIR = ROOT / "dist"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAGE_DEFINITIONS: List[dict] = [
    {
        "filename": "enhanced.html",
        "title": "🏈 NFL AI/ML Platform with Llama",
        "nav": {
            "brand": "NFL AI/ML Platform with Llama",
            "badge": "LLAMA POWERED",
            "links": [
                {"id": "metrics", "label": "Metrics"},
                {"id": "features", "label": "Capabilities"},
                {"id": "chat", "label": "AI Assistant"},
                {"id": "cta", "label": "Get access"},
            ],
        },
        "hero": {
            "heading": "Bring LLaMA intelligence to your touchdown predictions",
            "subheading": "Blend traditional analytics with retrieval-augmented reasoning and live roster context.",
            "description": "This enhanced experience ships with guarded LLaMA prompts, a curated knowledge base, and instant what-if analysis for coaches and scouts.",
            "cta": {"label": "Explore enhanced insights", "target": "features"},
        },
        "metrics": [
            {"icon": "fas fa-gauge", "label": "Latency", "value": "< 250 ms"},
            {"icon": "fas fa-chart-line", "label": "Touchdown lift", "value": "+9%"},
            {"icon": "fas fa-robot", "label": "LLM coverage", "value": "18K docs"},
            {"icon": "fas fa-users", "label": "Analyst seats", "value": "Unlimited"},
        ],
        "features": {
            "title": "Why teams choose the enhanced stack",
            "subtitle": "A single pane of glass for scouting, game planning, and RAG-assisted research.",
            "items": [
                {
                    "icon": "fas fa-layer-group",
                    "title": "Unified feature store",
                    "description": "Roll up rushing, passing, weather, and opponent metrics into reusable feature vectors synced across models.",
                },
                {
                    "icon": "fas fa-brain",
                    "title": "Context-aware LLaMA",
                    "description": "Ground large-language responses in verified Pinecone embeddings for trustworthy strategy notes.",
                },
                {
                    "icon": "fas fa-rocket",
                    "title": "Production hardening",
                    "description": "Observability hooks, structured logging, and rollout playbooks keep engineers in control week to week.",
                },
            ],
        },
        "ai_section": {
            "title": "AI assistant with retrieval reasoning",
            "detail_title": "Designed for analysts and coordinators",
            "detail_description": "Blend conversational queries with structured data pulls. Every answer ships with references so your coaching staff can trust the insight.",
            "highlights": [
                "Reference-linked answers sourced from Pinecone",
                "Confidence scoring with rationale snippets",
                "Drop-in Slack integration for sideline chats",
            ],
            "conversation": [
                {
                    "role": "user",
                    "text": "How does Mahomes perform against disguises on 3rd and long?",
                    "confidence": None,
                    "timestamp": "Coach",
                },
                {
                    "role": "ai",
                    "text": "Mahomes averages 8.4 YPA versus simulated pressure on 3rd and 7+. His touchdown probability climbs to 62%, driven by crossers. See full breakdown in the Chiefs defensive install doc.",
                    "confidence": 0.86,
                    "timestamp": "LLM",
                },
            ],
        },
        "call_to_action": {
            "title": "Upgrade your scouting workflow",
            "subtitle": "Schedule a walkthrough or clone the repository to run it locally.",
            "label": "Request a demo",
            "href": "mailto:hello@nfl-ml.example",
        },
        "footer": "Built by Shelton Bumhe — enhanced stack with LLaMA retrieval.",
        "theme": {
            "hero_start": "#1e3c72",
            "hero_end": "#2a5298",
            "user_message_bg": "#007bff",
            "ai_message_bg": "#ffffff",
            "positive": "#28a745",
            "negative": "#dc3545",
            "reasoning_border": "#007bff",
            "indicator_gradient": "linear-gradient(45deg,#ff6b6b,#4ecdc4)",
        },
        "extra_css": "",
    },
    {
        "filename": "production.html",
        "title": "🏈 NFL AI/ML Platform - Live Production",
        "nav": {
            "brand": "NFL AI/ML Platform — Production",
            "badge": "LIVE",
            "links": [
                {"id": "metrics", "label": "SLOs"},
                {"id": "features", "label": "Playbooks"},
                {"id": "chat", "label": "Operational AI"},
                {"id": "cta", "label": "Deploy"},
            ],
        },
        "hero": {
            "heading": "Serve touchdown predictions with production SLOs",
            "subheading": "Blueprinted for stadium-scale loads, blue/green releases, and compliance-ready logging.",
            "description": "Automated retraining, audit trails, and golden-path observability keep engineers confident on Sunday night.",
            "cta": {"label": "View deployment steps", "target": "cta"},
        },
        "metrics": [
            {"icon": "fas fa-clock", "label": "Release cadence", "value": "Weekly"},
            {"icon": "fas fa-cloud", "label": "Cloud targets", "value": "AWS / GCP"},
            {"icon": "fas fa-database", "label": "Data retention", "value": "180 days"},
            {"icon": "fas fa-shield-halved", "label": "Runtime policies", "value": "OPA"},
        ],
        "features": {
            "title": "Production playbooks",
            "subtitle": "Everything required to harden the stack for fans, broadcasters, and sportsbooks.",
            "items": [
                {
                    "icon": "fas fa-diagram-project",
                    "title": "Orchestrated workflows",
                    "description": "GitHub Actions deploys to Render, Railway, or ECS with health checks and migration gates.",
                },
                {
                    "icon": "fas fa-heart-pulse",
                    "title": "Runtime observability",
                    "description": "Ship Prometheus metrics, structured logs, and OpenTelemetry traces out of the box.",
                },
                {
                    "icon": "fas fa-user-shield",
                    "title": "Policy guardrails",
                    "description": "Secret scanning, least privilege IAM, and adaptive rate limits for partner APIs.",
                },
            ],
        },
        "ai_section": {
            "title": "Operations co-pilot",
            "detail_title": "24/7 incident ready",
            "detail_description": "Ask the assistant for rollback steps, data drift reports, or release validations.",
            "highlights": [
                "Incident runbooks with links into observability dashboards",
                "Auto-summarised change logs for coaching staff",
                "Assistive prompts for real-time anomaly triage",
            ],
            "conversation": [
                {"role": "user", "text": "Why did latency spike in the 4th quarter?", "confidence": None, "timestamp": "SRE"},
                {"role": "ai", "text": "Live inference hit autoscale limits. Queue depth reached 80th percentile; launching two additional instances resolved latency. See Grafana dashboard 12.", "confidence": 0.78, "timestamp": "AI"},
            ],
        },
        "call_to_action": {
            "title": "Launch the production stack",
            "subtitle": "Use docker-compose locally, then promote via Render, Railway, or ECS pipelines.",
            "label": "Open deployment guide",
            "href": "../DEPLOYMENT_GUIDE.md",
        },
        "footer": "Production deployment kit — instrumentation ready.",
        "theme": {
            "hero_start": "#2d5a27",
            "hero_end": "#4a7c59",
            "user_message_bg": "#2d5a27",
            "ai_message_bg": "#ffffff",
            "positive": "#28a745",
            "negative": "#dc3545",
            "reasoning_border": "#2d5a27",
            "indicator_gradient": "linear-gradient(45deg,#2d5a27,#4a7c59,#6b8e23)",
        },
        "extra_css": "",
    },
    {
        "filename": "comprehensive.html",
        "title": "🏈 NFL AI/ML Platform - Comprehensive",
        "nav": {
            "brand": "NFL AI/ML Platform",
            "badge": "KNOWLEDGE GRAPH",
            "links": [
                {"id": "metrics", "label": "Coverage"},
                {"id": "features", "label": "Modules"},
                {"id": "chat", "label": "Knowledge hub"},
                {"id": "cta", "label": "Download"},
            ],
        },
        "hero": {
            "heading": "Curate a complete football knowledge graph",
            "subheading": "Join structured stats, scouting notes, play diagrams, and contract data across the league.",
            "description": "Comprehensive mode focuses on data stewardship—perfect for analysts building season-long dashboards.",
            "cta": {"label": "Browse modules", "target": "features"},
        },
        "metrics": [
            {"icon": "fas fa-database", "label": "Tables", "value": "16"},
            {"icon": "fas fa-file-import", "label": "Seed scripts", "value": "8"},
            {"icon": "fas fa-book", "label": "Play concepts", "value": "240+"},
            {"icon": "fas fa-lightbulb", "label": "Knowledge cards", "value": "1.3K"},
        ],
        "features": {
            "title": "Cross-cutting modules",
            "subtitle": "Mix scouting, medical, contract, and tracking data in a governed workspace.",
            "items": [
                {
                    "icon": "fas fa-share-nodes",
                    "title": "Linked entities",
                    "description": "Players, teams, games, and seasons remain connected through SQLAlchemy models and seed notebooks.",
                },
                {
                    "icon": "fas fa-sitemap",
                    "title": "Dimensional playbooks",
                    "description": "Organise plays by formation, coverage, and success rate—perfect for opponent scouting.",
                },
                {
                    "icon": "fas fa-cloud-arrow-down",
                    "title": "Bulk ingestion",
                    "description": "Automated loaders validate CSVs, enforce schema constraints, and populate SQLite or Postgres.",
                },
            ],
        },
        "ai_section": {
            "title": "Knowledge chat",
            "detail_title": "Guided discovery",
            "detail_description": "Surface insights from the knowledge graph with citations back to the original dataset.",
            "highlights": [
                "Drill into play-by-play tendencies with filters",
                "Summaries cite document IDs and timestamps",
                "Supports natural language or SQL-like prompts",
            ],
            "conversation": [
                {"role": "user", "text": "Give me chunk plays from Shanahan’s offenses vs Cover 3.", "confidence": None, "timestamp": "Analyst"},
                {"role": "ai", "text": "Found 27 explosive plays. 63% came from outside zone variations. Linked report: 2023-49ers-explosives.pdf", "confidence": 0.81, "timestamp": "Assistant"},
            ],
        },
        "call_to_action": {
            "title": "Build your football knowledge base",
            "subtitle": "Clone the repository and run the seed scripts or request a turnkey deployment.",
            "label": "Clone on GitHub",
            "href": "https://github.com/sheltonbumhe/machine-learning-nfl-touchdowns",
        },
        "footer": "Comprehensive knowledge graph edition.",
        "theme": {
            "hero_start": "#2d5a27",
            "hero_end": "#4a7c59",
            "user_message_bg": "#2d5a27",
            "ai_message_bg": "#ffffff",
            "positive": "#28a745",
            "negative": "#dc3545",
            "reasoning_border": "#2d5a27",
            "indicator_gradient": "linear-gradient(45deg,#2d5a27,#4a7c59,#6b8e23)",
        },
        "extra_css": "",
    },
    {
        "filename": "simple.html",
        "title": "🏈 NFL AI/ML Platform",
        "nav": {
            "brand": "NFL AI/ML Platform",
            "badge": None,
            "links": [
                {"id": "metrics", "label": "Highlights"},
                {"id": "features", "label": "Features"},
                {"id": "chat", "label": "Assistant"},
                {"id": "cta", "label": "Get started"},
            ],
        },
        "hero": {
            "heading": "Launch touchdown predictions in an afternoon",
            "subheading": "The streamlined distribution for hackathons, demos, and quick proofs of concept.",
            "description": "Spin up SQLite, run the orchestrator, and experiment with real NFL data without wrestling with cloud infrastructure.",
            "cta": {"label": "Run locally", "target": "cta"},
        },
        "metrics": [
            {"icon": "fas fa-bolt", "label": "Setup time", "value": "< 15 min"},
            {"icon": "fas fa-database", "label": "Bundled CSVs", "value": "3"},
            {"icon": "fas fa-brain", "label": "Model accuracy", "value": "88%"},
            {"icon": "fas fa-user-check", "label": "Demo users", "value": "Unlimited"},
        ],
        "features": {
            "title": "Simple mode highlights",
            "subtitle": "Perfect for portfolio demos and workshops.",
            "items": [
                {
                    "icon": "fas fa-play-circle",
                    "title": "One command workflow",
                    "description": "`python main.py` loads data, validates quality, trains the model, and prints status.",
                },
                {
                    "icon": "fas fa-laptop-code",
                    "title": "Streamlined UI",
                    "description": "Interact with predictions via CLI or upgrade to the React dashboard when ready.",
                },
                {
                    "icon": "fas fa-book-open",
                    "title": "Educational notebooks",
                    "description": "Walkthrough notebooks cover EDA, feature engineering, and explainability with SHAP.",
                },
            ],
        },
        "ai_section": {
            "title": "Assistant-included",
            "detail_title": "Help at every step",
            "detail_description": "Use the assistant to explain model outputs, explore data drift, or suggest new features.",
            "highlights": [
                "Explain each prediction in plain language",
                "Generate feature engineering snippets",
                "Share quick tips for first-time users",
            ],
            "conversation": [
                {"role": "user", "text": "Why did the model pick a touchdown here?", "confidence": None, "timestamp": "Builder"},
                {"role": "ai", "text": "High rolling yards and red-zone efficiency push probability to 81%. Consider monitoring interception rate drift this week.", "confidence": 0.8, "timestamp": "Assistant"},
            ],
        },
        "call_to_action": {
            "title": "Clone and run today",
            "subtitle": "Install Python 3.9+, run `pip install -r requirements.txt`, and launch the workflow.",
            "label": "View quickstart",
            "href": "../README.md",
        },
        "footer": "Simple starter edition for analysts and students.",
        "theme": {
            "hero_start": "#1e3c72",
            "hero_end": "#2a5298",
            "user_message_bg": "#e3f2fd",
            "ai_message_bg": "#f5f5f5",
            "positive": "#28a745",
            "negative": "#dc3545",
            "reasoning_border": "#1e3c72",
            "indicator_gradient": "linear-gradient(45deg,#1e3c72,#2a5298)",
        },
        "extra_css": "",
    },
    {
        "filename": "ultimate.html",
        "title": "🏈 NFL AI Platform - Ultimate Production",
        "nav": {
            "brand": "NFL AI Platform",
            "badge": "ULTIMATE",
            "links": [
                {"id": "metrics", "label": "Scale"},
                {"id": "features", "label": "Modules"},
                {"id": "chat", "label": "Ops AI"},
                {"id": "cta", "label": "Engage"},
            ],
        },
        "hero": {
            "heading": "Operate the league-wide analytics control plane",
            "subheading": "Multi-tenant, compliance-ready, and hardened for broadcast partners.",
            "description": "Ultimate mode bundles Kubernetes manifests, MySQL + Redis replication, and advanced access controls.",
            "cta": {"label": "Review architecture", "target": "metrics"},
        },
        "metrics": [
            {"icon": "fas fa-server", "label": "Regions", "value": "3"},
            {"icon": "fas fa-users-cog", "label": "Role profiles", "value": "12"},
            {"icon": "fas fa-file-shield", "label": "Compliance", "value": "SOC2-ready"},
            {"icon": "fas fa-plug", "label": "Integrations", "value": "15"},
        ],
        "features": {
            "title": "Enterprise modules",
            "subtitle": "Run analytics for leagues, media partners, and sportsbooks.",
            "items": [
                {
                    "icon": "fas fa-key",
                    "title": "Fine-grained access",
                    "description": "Policy-based access per organization with audit logging and SSO hooks.",
                },
                {
                    "icon": "fas fa-cloud",
                    "title": "Hybrid deploy",
                    "description": "Support Kubernetes, ECS, or bare-metal clusters with identical IaC playbooks.",
                },
                {
                    "icon": "fas fa-shield",
                    "title": "Resilience",
                    "description": "Chaos testing scenarios, circuit breakers, and fallback heuristics keep latency predictable.",
                },
            ],
        },
        "ai_section": {
            "title": "Ops assistant",
            "detail_title": "Enterprise-aware",
            "detail_description": "Stay ahead of audits, contract renewals, and critical incidents with context-rich responses.",
            "highlights": [
                "Surface SLAs and partner agreements",
                "Summarise compliance posture with evidence",
                "Suggest scaling actions during peak demand",
            ],
            "conversation": [
                {"role": "user", "text": "Prepare talking points for the rights-holder briefing.", "confidence": None, "timestamp": "Exec"},
                {"role": "ai", "text": "Key wins: 99.95% uptime across three regions, 18% uplift in engagement, new data products shipped in 6 weeks.", "confidence": 0.9, "timestamp": "Assistant"},
            ],
        },
        "call_to_action": {
            "title": "Schedule an architectural review",
            "subtitle": "Walk through tenancy, compliance, and reliability requirements with the engineering team.",
            "label": "Book a meeting",
            "href": "mailto:enterprise@nfl-ml.example",
        },
        "footer": "Ultimate enterprise edition — SLAs included.",
        "theme": {
            "hero_start": "#0d4f3c",
            "hero_end": "#2d8f6b",
            "user_message_bg": "#0d4f3c",
            "ai_message_bg": "#ffffff",
            "positive": "#28a745",
            "negative": "#dc3545",
            "reasoning_border": "#0d4f3c",
            "indicator_gradient": "linear-gradient(45deg,#0d4f3c,#1a7a5e,#2d8f6b)",
        },
        "extra_css": "",
    },
]


def main():
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template('template.html')

    for page in PAGE_DEFINITIONS:
        output_path = OUTPUT_DIR / page["filename"]
        html = template.render(page=page)
        output_path.write_text(html)
        print(f"Generated {output_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
