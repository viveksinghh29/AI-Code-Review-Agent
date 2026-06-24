"""
GitHub Integration Module
==========================
Connects the AI Code Review Agent to GitHub APIs via PyGithub.

Responsibilities:
  - Authenticate with a GitHub personal access token
  - Fetch repository metadata and open pull requests
  - Post inline review comments on pull request files
  - Create PR-level summary comments
  - Generate formatted markdown PR summaries

Classes:
  GitHubClient        — authenticated PyGithub wrapper (lazy init)
  PRFetcher           — fetches PR metadata + changed files
  PRReviewPoster      — posts inline + summary comments on a PR
  GitHubIntegration   — public facade consumed by Orchestrator / Dashboard
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from backend.reviewer.ai_reviewer import ReviewComment, ReviewReport
from backend.utils.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PRFile:
    """A single file changed in a pull request."""
    filename:  str
    status:    str      # added | modified | removed | renamed
    additions: int
    deletions: int
    changes:   int
    patch:     str      # unified diff patch text


@dataclass
class PullRequestInfo:
    """Metadata for one pull request."""
    number:      int
    title:       str
    author:      str
    state:       str    # open | closed | merged
    base_branch: str
    head_branch: str
    url:         str
    created_at:  str
    updated_at:  str
    body:        str
    files:       list[PRFile] = field(default_factory=list)
    labels:      list[str]    = field(default_factory=list)

    @property
    def short_title(self) -> str:
        return self.title[:80]


@dataclass
class PostedComment:
    """A review comment successfully posted to GitHub."""
    comment_id:  int
    file_path:   str
    line_number: int
    body:        str
    url:         str


@dataclass
class PRReviewResult:
    """Result of posting a full review to a pull request."""
    pr_number:           int
    repo_full_name:      str
    posted_comments:     list[PostedComment]
    summary_comment_id:  Optional[int]
    errors:              list[str]

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_posted(self) -> int:
        return len(self.posted_comments)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class GitHubIntegrationError(Exception):
    pass

class NotAuthenticatedError(GitHubIntegrationError):
    pass

class RepositoryNotFoundError(GitHubIntegrationError):
    pass

class PRNotFoundError(GitHubIntegrationError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# GitHub Client
# ─────────────────────────────────────────────────────────────────────────────

class GitHubClient:
    """Lazy, authenticated PyGithub wrapper."""

    def __init__(self, token: Optional[str] = None):
        self._token  = token or get_config().github.token
        self._github = None

    def _get_github(self):
        if self._github is not None:
            return self._github
        if not self._token:
            raise NotAuthenticatedError(
                "No GitHub token configured. "
                "Set GITHUB_TOKEN in your .env file.\n"
                "Create a token at: https://github.com/settings/tokens"
            )
        try:
            from github import Github
        except ImportError:
            raise GitHubIntegrationError(
                "PyGithub not installed. Run: pip install PyGithub"
            )
        self._github = Github(self._token)
        return self._github

    def get_repo(self, full_name: str):
        try:
            return self._get_github().get_repo(full_name)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                raise RepositoryNotFoundError(
                    f"Repository '{full_name}' not found. "
                    "Check the name and ensure your token has access."
                ) from exc
            raise GitHubIntegrationError(
                f"Failed to fetch repository '{full_name}': {exc}"
            ) from exc

    @staticmethod
    def extract_full_name(url: str) -> str:
        """Extract 'owner/repo' from any GitHub URL format."""
        url = url.strip().rstrip("/").removesuffix(".git")
        # Require github.com preceded by // or @ to avoid matching subdomains
        m = re.search(
            r"(?:^|[/@:])github\.com[:/](?P<full>[\w\-\.]+/[\w\-\.]+)",
            url,
        )
        if m:
            return m.group("full")
        raise GitHubIntegrationError(
            f"Cannot extract owner/repo from URL: {url!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PR Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class PRFetcher:
    """Fetches PR metadata and changed files from GitHub."""

    def __init__(self, client: GitHubClient):
        self._client = client

    def list_open_prs(self, repo_url: str) -> list[PullRequestInfo]:
        """Return up to 30 open PRs for a repository URL."""
        full_name = self._client.extract_full_name(repo_url)
        repo      = self._client.get_repo(full_name)
        prs: list[PullRequestInfo] = []
        try:
            for pr in repo.get_pulls(state="open", sort="updated",
                                     direction="desc")[:30]:
                prs.append(self._to_pr_info(pr, include_files=False))
        except Exception as exc:
            logger.warning(f"Could not list PRs for {full_name}: {exc}")
        logger.info(f"Found {len(prs)} open PRs for {full_name}")
        return prs

    def get_pr(self, repo_url: str, pr_number: int) -> PullRequestInfo:
        """Fetch full metadata + changed files for one PR."""
        full_name = self._client.extract_full_name(repo_url)
        repo      = self._client.get_repo(full_name)
        try:
            pr = repo.get_pull(pr_number)
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "404" in msg:
                raise PRNotFoundError(
                    f"PR #{pr_number} not found in {full_name}."
                ) from exc
            raise GitHubIntegrationError(
                f"Failed to fetch PR #{pr_number}: {exc}"
            ) from exc
        return self._to_pr_info(pr, include_files=True)

    def _to_pr_info(self, pr, include_files: bool) -> PullRequestInfo:
        files: list[PRFile] = []
        if include_files:
            try:
                for f in pr.get_files():
                    files.append(PRFile(
                        filename  = f.filename,
                        status    = f.status,
                        additions = f.additions,
                        deletions = f.deletions,
                        changes   = f.changes,
                        patch     = f.patch or "",
                    ))
            except Exception as exc:
                logger.warning(
                    f"Could not fetch files for PR #{pr.number}: {exc}"
                )
        labels = []
        try:
            labels = [label.name for label in pr.labels]
        except Exception:
            pass

        return PullRequestInfo(
            number      = pr.number,
            title       = pr.title,
            author      = pr.user.login if pr.user else "unknown",
            state       = pr.state,
            base_branch = pr.base.ref,
            head_branch = pr.head.ref,
            url         = pr.html_url,
            created_at  = pr.created_at.isoformat() if pr.created_at else "",
            updated_at  = pr.updated_at.isoformat() if pr.updated_at else "",
            body        = (pr.body or "")[:1000],
            files       = files,
            labels      = labels,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PR Review Poster
# ─────────────────────────────────────────────────────────────────────────────

class PRReviewPoster:
    """Posts AI-generated review comments to a GitHub pull request."""

    MIN_SEVERITY_TO_POST = {"Critical", "High", "Medium"}

    def __init__(self, client: GitHubClient):
        self._client = client

    def post_review(
        self,
        repo_url:     str,
        pr_number:    int,
        pr_info:      PullRequestInfo,
        review:       ReviewReport,
        min_severity: Optional[set[str]] = None,
        dry_run:      bool = False,
    ) -> PRReviewResult:
        """
        Post review comments to a GitHub pull request.
        With dry_run=True, simulates all actions without API calls.
        """
        min_sev   = min_severity or self.MIN_SEVERITY_TO_POST
        full_name = self._client.extract_full_name(repo_url)

        pr = None
        if not dry_run:
            repo = self._client.get_repo(full_name)
            try:
                pr = repo.get_pull(pr_number)
            except Exception as exc:
                raise PRNotFoundError(
                    f"PR #{pr_number} not found in {full_name}."
                ) from exc

        changed_files   = {f.filename for f in pr_info.files}
        posted_comments: list[PostedComment] = []
        errors:          list[str]           = []

        # ── Inline comments ─────────────────────────────────────────────────
        for file_review in review.file_reviews:
            if file_review.file_name not in changed_files:
                continue
            for comment in file_review.comments:
                if comment.severity not in min_sev:
                    continue
                body = self._format_inline_comment(comment)

                if dry_run:
                    posted_comments.append(PostedComment(
                        comment_id  = -1,
                        file_path   = comment.file_name,
                        line_number = comment.line_number,
                        body        = body,
                        url         = f"dry-run://{full_name}/pull/{pr_number}",
                    ))
                    continue

                try:
                    head_commit = pr.get_commits().reversed[0]
                    gh_comment  = pr.create_review_comment(
                        body   = body,
                        commit = head_commit,
                        path   = comment.file_name,
                        line   = comment.line_number,
                    )
                    posted_comments.append(PostedComment(
                        comment_id  = gh_comment.id,
                        file_path   = comment.file_name,
                        line_number = comment.line_number,
                        body        = body,
                        url         = gh_comment.html_url,
                    ))
                    logger.info(
                        f"Posted inline comment: "
                        f"{comment.file_name}:{comment.line_number}"
                    )
                except Exception as exc:
                    err = (
                        f"Could not post inline comment on "
                        f"{comment.file_name}:{comment.line_number}: {exc}"
                    )
                    errors.append(err)
                    logger.warning(err)

        # ── Summary comment ─────────────────────────────────────────────────
        summary_comment_id: Optional[int] = None
        summary_body = self._format_summary_comment(review, full_name, pr_number)

        if dry_run:
            summary_comment_id = -1
        else:
            try:
                gh_summary         = pr.create_issue_comment(summary_body)
                summary_comment_id = gh_summary.id
                logger.info(
                    f"Posted summary comment on PR #{pr_number} "
                    f"(id={summary_comment_id})"
                )
            except Exception as exc:
                err = f"Could not post summary comment: {exc}"
                errors.append(err)
                logger.error(err)

        result = PRReviewResult(
            pr_number          = pr_number,
            repo_full_name     = full_name,
            posted_comments    = posted_comments,
            summary_comment_id = summary_comment_id,
            errors             = errors,
        )
        logger.info(
            f"PR review posted: {result.total_posted} inline, "
            f"{'1' if summary_comment_id else '0'} summary, "
            f"{len(errors)} errors"
        )
        return result

    # ── Formatting helpers ────────────────────────────────────────────────────

    @staticmethod
    def _format_inline_comment(comment: ReviewComment) -> str:
        sev_emoji = {
            "Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"
        }.get(comment.severity, "⚪")
        cat_emoji = {
            "Security":"🔒","Performance":"⚡","Bug Risk":"🐛",
            "Maintainability":"🔧","Readability":"📖",
            "Best Practices":"✅","Scalability":"📈",
        }.get(comment.category, "📌")
        src = "AST Analysis" if comment.is_ast_detected else "AI Review"

        lines = [
            f"## {sev_emoji} {comment.severity} — `{comment.issue_type}`",
            f"",
            f"**{cat_emoji} Category:** {comment.category} &nbsp;|&nbsp; "
            f"**Confidence:** {comment.confidence_score}% &nbsp;|&nbsp; "
            f"**Source:** {src}",
            f"",
            f"### Problem",
            comment.explanation,
        ]
        if comment.suggested_fix:
            lines += [
                f"",
                f"### Suggested Fix",
                f"```python",
                comment.suggested_fix,
                f"```",
            ]
        lines += [
            f"",
            f"---",
            f"<sub>🤖 Generated by AI Code Review Agent</sub>",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_summary_comment(
        review: ReviewReport, repo_full_name: str, pr_number: int
    ) -> str:
        lines = [
            f"# 🔍 AI Code Review Summary",
            f"",
            f"> **Repository:** `{repo_full_name}` &nbsp;|&nbsp; "
            f"**PR:** #{pr_number} &nbsp;|&nbsp; "
            f"**Provider:** {review.llm_provider} / `{review.llm_model}`",
            f"",
            f"## Results",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Files reviewed | {len(review.file_reviews)} |",
            f"| Total comments | {review.total_comments} |",
            f"| 🔴 Critical | {review.critical_count} |",
            f"| 🟠 High | {review.high_count} |",
            f"| 🟡 Medium | {review.medium_count} |",
            f"| 🟢 Low | {review.low_count} |",
            f"| Avg quality score | {review.avg_quality_score:.0f}/100 |",
            f"| Avg confidence | {review.avg_confidence:.0f}% |",
            f"",
        ]
        critical_high = [
            c for c in review.all_comments()
            if c.severity in ("Critical", "High")
        ][:10]
        if critical_high:
            lines += ["## 🚨 Critical & High Priority Findings", ""]
            for c in critical_high:
                icon = "🔴" if c.severity == "Critical" else "🟠"
                lines.append(
                    f"- {icon} **`{c.file_name}`** L{c.line_number} "
                    f"[`{c.issue_type}`] — {c.explanation[:100]}"
                )
            lines.append("")
        lines += [
            f"---",
            f"<sub>🤖 Generated by AI Code Review Agent · "
            f"{review.llm_provider}/{review.llm_model}</sub>",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public Facade
# ─────────────────────────────────────────────────────────────────────────────

class GitHubIntegration:
    """
    Public facade for all GitHub integration features.

    Usage:
        gh  = GitHubIntegration()
        prs = gh.list_open_prs("https://github.com/owner/repo")
        res = gh.post_pr_review(
                  repo_url="https://github.com/owner/repo",
                  pr_number=42,
                  review=review_report,
                  dry_run=True,
              )
    """

    def __init__(self, token: Optional[str] = None):
        self._client  = GitHubClient(token)
        self._fetcher = PRFetcher(self._client)
        self._poster  = PRReviewPoster(self._client)

    @property
    def is_authenticated(self) -> bool:
        return bool(self._client._token)

    def list_open_prs(self, repo_url: str) -> list[PullRequestInfo]:
        return self._fetcher.list_open_prs(repo_url)

    def get_pr(self, repo_url: str, pr_number: int) -> PullRequestInfo:
        return self._fetcher.get_pr(repo_url, pr_number)

    def post_pr_review(
        self,
        repo_url:     str,
        pr_number:    int,
        review:       ReviewReport,
        min_severity: Optional[set[str]] = None,
        dry_run:      bool = False,
    ) -> PRReviewResult:
        pr_info = self._fetcher.get_pr(repo_url, pr_number)
        return self._poster.post_review(
            repo_url     = repo_url,
            pr_number    = pr_number,
            pr_info      = pr_info,
            review       = review,
            min_severity = min_severity,
            dry_run      = dry_run,
        )

    def generate_pr_summary(
        self, review: ReviewReport, repo_url: str, pr_number: int
    ) -> str:
        """Generate Markdown summary without posting it."""
        full_name = self._client.extract_full_name(repo_url)
        return self._poster._format_summary_comment(review, full_name, pr_number)
