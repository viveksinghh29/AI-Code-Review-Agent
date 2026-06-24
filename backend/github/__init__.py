"""GitHub integration package — PR fetching, posting, and summaries."""

from backend.github.github_integration import (
    GitHubIntegration,
    GitHubClient,
    PRFetcher,
    PRReviewPoster,
    PullRequestInfo,
    PRFile,
    PostedComment,
    PRReviewResult,
    GitHubIntegrationError,
    NotAuthenticatedError,
    RepositoryNotFoundError,
    PRNotFoundError,
)

__all__ = [
    "GitHubIntegration",
    "GitHubClient",
    "PRFetcher",
    "PRReviewPoster",
    "PullRequestInfo",
    "PRFile",
    "PostedComment",
    "PRReviewResult",
    "GitHubIntegrationError",
    "NotAuthenticatedError",
    "RepositoryNotFoundError",
    "PRNotFoundError",
]
