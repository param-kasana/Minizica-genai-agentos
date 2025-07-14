import asyncio
import os
from typing import Annotated, Optional, List
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

from dotenv import load_dotenv
load_dotenv()
from github import Github
from concurrent.futures import ThreadPoolExecutor

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMTk2YzA3MS00ZjRkLTQ1MWQtOTBjZi0yZGM3MDE5ZWNlMzIiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6ImI5OWQ4YTNmLWFjNTItNGRmNC05YjExLTYxNzljZDQ4Y2MxNSJ9.Nhq3znLf7a20Is-ULhufUOUgXcxkaekv0bLT_5DRT44" 
session = GenAISession(jwt_token=AGENT_JWT)
_executor = ThreadPoolExecutor()

class GitHubActionsAgent:
    def __init__(self):
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        self.g = Github(github_token)
        self.user = self.g.get_user().login

    # 1. Get basic repo info
    def get_repo(self, repo_full_name):
        repo = self.g.get_repo(repo_full_name)
        return {
            "name": repo.name,
            "full_name": repo.full_name,
            "description": repo.description
        }

    # 2. List all branches
    def list_branches(self, repo_full_name):
        repo = self.g.get_repo(repo_full_name)
        return [branch.name for branch in repo.get_branches()]

    # 3. Get info on a single branch
    def get_branch_info(self, repo_full_name, branch_name):
        repo = self.g.get_repo(repo_full_name)
        branch = repo.get_branch(branch_name)
        return {
            "name": branch.name,
            "commit_sha": branch.commit.sha,
            "protected": branch.protected
        }

    # 4. List all items in a directory (non-recursive, no content)
    def list_directory_contents(self, repo_full_name, directory_path="", branch="main"):
        repo = self.g.get_repo(repo_full_name)
        return [
            {
                "path": content.path,
                "type": content.type,
                "sha": content.sha
            }
            for content in repo.get_contents(directory_path, ref=branch)
        ]

    # 5. Fetch content of a single file (by path)
    def fetch_file_content(self, repo_full_name, file_path, branch="main"):
        repo = self.g.get_repo(repo_full_name)
        file_content = repo.get_contents(file_path, ref=branch)
        return {
            "content": file_content.decoded_content.decode(),
            "sha": file_content.sha,
            "path": file_content.path
        }

    # 6. Recursively get all files+contents in repo/branch
    def fetch_all_files_recursive(self, repo_full_name, branch="main"):
        repo = self.g.get_repo(repo_full_name)
        files = []

        def _traverse(path=""):
            contents = repo.get_contents(path, ref=branch)
            for content in contents:
                if content.type == "dir":
                    _traverse(content.path)
                else:
                    files.append({
                        "path": content.path,
                        "content": content.decoded_content.decode(),
                        "sha": content.sha
                    })

        _traverse()
        return files

    # 7. Create new file
    def create_file(self, repo_full_name, file_path, content, commit_message, branch="main"):
        repo = self.g.get_repo(repo_full_name)
        result = repo.create_file(file_path, commit_message, content, branch=branch)
        content_obj = result.get("content", None)
        commit_obj = result.get("commit", None)
        return {
            "path": getattr(content_obj, "path", None) if content_obj else None,
            "sha": getattr(content_obj, "sha", None) if content_obj else None,
            "commit_url": getattr(commit_obj, "html_url", None) if commit_obj else None,
            "commit_message": getattr(commit_obj.commit, "message", None) if (commit_obj and hasattr(commit_obj, "commit")) else None
        }

    # 8. Update file
    def update_file(self, repo_full_name, file_path, branch, new_content, sha, commit_message):
        repo = self.g.get_repo(repo_full_name)
        result = repo.update_file(
            file_path,
            commit_message,
            new_content,
            sha,
            branch=branch
        )
        content_obj = result.get("content", None)
        commit_obj = result.get("commit", None)
        return {
            "path": getattr(content_obj, "path", None) if content_obj else None,
            "sha": getattr(content_obj, "sha", None) if content_obj else None,
            "commit_url": getattr(commit_obj, "html_url", None) if commit_obj else None,
            "commit_message": getattr(commit_obj.commit, "message", None) if (commit_obj and hasattr(commit_obj, "commit")) else None
        }

    # 9. Delete file
    def delete_file(self, repo_full_name, file_path, sha, commit_message, branch="main"):
        repo = self.g.get_repo(repo_full_name)
        result = repo.delete_file(
            file_path,
            commit_message,
            sha,
            branch=branch
        )
        commit_obj = result.get("commit", None)
        return {
            "commit_url": getattr(commit_obj, "html_url", None) if commit_obj else None,
            "commit_message": getattr(commit_obj.commit, "message", None) if (commit_obj and hasattr(commit_obj, "commit")) else None
        }

    # 10. Create a branch from source branch
    def create_branch(self, repo_full_name, source_branch, new_branch):
        repo = self.g.get_repo(repo_full_name)
        source = repo.get_branch(source_branch)
        repo.create_git_ref(ref='refs/heads/' + new_branch, sha=source.commit.sha)

    # 11. Create pull request
    def create_pull_request(self, repo_full_name, head_branch, base_branch, title, body):
        repo = self.g.get_repo(repo_full_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch
        )
        return {
            "number": pr.number,
            "title": pr.title,
            "url": pr.html_url,
            "state": pr.state
        }

    # 12. Add reviewers to a pull request
    def add_reviewers_to_pr(self, repo_full_name, pr_number, reviewers):
        repo = self.g.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        pr.create_review_request(reviewers=reviewers)
        return {"status": "reviewers added", "reviewers": reviewers}

    # 13. Add comment to PR
    def add_pr_comment(self, repo_full_name, pr_number, comment_body):
        repo = self.g.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        comment = pr.create_issue_comment(comment_body)
        return {"comment_id": comment.id, "body": comment.body}

    # 14. Edit PR comment
    def edit_pr_comment(self, repo_full_name, comment_id, new_body):
        repo = self.g.get_repo(repo_full_name)
        comment = repo.get_issue_comment(comment_id)
        comment.edit(new_body)
        return {"comment_id": comment.id, "body": comment.body}

    # 15. List PRs assigned to me
    def list_prs_assigned_to_me(self, repo_full_name):
        repo = self.g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state="open")
        assigned_prs = []
        for pr in pulls:
            reviewers = [r.login for r in pr.get_review_requests()[0]]
            if self.user in reviewers:
                assigned_prs.append({
                    "number": pr.number,
                    "title": pr.title,
                    "url": pr.html_url,
                    "author": pr.user.login
                })
        return assigned_prs

    # 16. List PRs created by me
    def list_prs_created_by_me(self, repo_full_name):
        repo = self.g.get_repo(repo_full_name)
        pulls = repo.get_pulls(state="open")
        my_prs = []
        for pr in pulls:
            if pr.user.login == self.user:
                my_prs.append({
                    "number": pr.number,
                    "title": pr.title,
                    "url": pr.html_url,
                    "author": pr.user.login
                })
        return my_prs

agent = GitHubActionsAgent()

def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: func(*args, **kwargs))

@session.bind(
    name="github_agent",
    description="Comprehensive GitHub agent: repo info, file/folder listing, file content fetch, CRUD operations, branch/PR management, and comments/reviews."
)
async def github_agent(
    agent_context: GenAIContext,
    action: Annotated[
        str,
        "Action to perform: get_repo, list_branches, get_branch_info, list_directory_contents, fetch_file_content, fetch_all_files_recursive, create_file, update_file, delete_file, create_branch, create_pull_request, add_reviewers_to_pr, add_pr_comment, edit_pr_comment, list_prs_assigned_to_me, list_prs_created_by_me"
    ],
    repo_full_name: Annotated[
        str,
        "Full repo name, e.g. 'owner/repo'"
    ],
    directory_path: Annotated[
        Optional[str],
        "Directory path for listing files/folders (for list_directory_contents). Leave blank for root."
    ] = "",
    file_path: Annotated[
        Optional[str],
        "Path to file (for fetch_file_content, create_file, update_file, delete_file)."
    ] = None,
    branch: Annotated[
        Optional[str],
        "Branch name (default: main, applies to most actions)."
    ] = "main",
    source_branch: Annotated[
        Optional[str],
        "Source branch (for create_branch)."
    ] = None,
    new_branch: Annotated[
        Optional[str],
        "New branch name (for create_branch)."
    ] = None,
    new_content: Annotated[
        Optional[str],
        "New file content (for update_file)."
    ] = None,
    sha: Annotated[
        Optional[str],
        "SHA of file (for update_file, delete_file)."
    ] = None,
    commit_message: Annotated[
        Optional[str],
        "Commit message (for update_file, create_file, delete_file)."
    ] = None,
    content: Annotated[
        Optional[str],
        "Content for create_file."
    ] = None,
    head_branch: Annotated[
        Optional[str],
        "Head branch (for create_pull_request)."
    ] = None,
    base_branch: Annotated[
        Optional[str],
        "Base branch (for create_pull_request)."
    ] = None,
    title: Annotated[
        Optional[str],
        "Title (for create_pull_request)."
    ] = None,
    body: Annotated[
        Optional[str],
        "Body (for create_pull_request, add_pr_comment, edit_pr_comment)."
    ] = None,
    pr_number: Annotated[
        Optional[int],
        "Pull request number (for add_reviewers_to_pr, add_pr_comment)."
    ] = None,
    reviewers: Annotated[
        Optional[List[str]],
        "List of GitHub usernames to add as reviewers (for add_reviewers_to_pr)."
    ] = None,
    branch_name: Annotated[
        Optional[str],
        "Branch name (for get_branch_info)."
    ] = None,
    comment_id: Annotated[
        Optional[int],
        "Comment ID (for edit_pr_comment)."
    ] = None,
    new_body: Annotated[
        Optional[str],
        "New comment body (for edit_pr_comment)."
    ] = None,
):
    try:
        if action == "get_repo":
            return await run_in_executor(agent.get_repo, repo_full_name)
        elif action == "list_branches":
            return {"branches": await run_in_executor(agent.list_branches, repo_full_name)}
        elif action == "get_branch_info":
            return await run_in_executor(agent.get_branch_info, repo_full_name, branch_name)
        elif action == "list_directory_contents":
            return {"contents": await run_in_executor(agent.list_directory_contents, repo_full_name, directory_path, branch)}
        elif action == "fetch_file_content":
            return await run_in_executor(agent.fetch_file_content, repo_full_name, file_path, branch)
        elif action == "fetch_all_files_recursive":
            return {"files": await run_in_executor(agent.fetch_all_files_recursive, repo_full_name, branch)}
        elif action == "create_file":
            return await run_in_executor(agent.create_file, repo_full_name, file_path, content, commit_message, branch)
        elif action == "update_file":
            return await run_in_executor(agent.update_file, repo_full_name, file_path, branch, new_content, sha, commit_message)
        elif action == "delete_file":
            return await run_in_executor(agent.delete_file, repo_full_name, file_path, sha, commit_message, branch)
        elif action == "create_branch":
            await run_in_executor(agent.create_branch, repo_full_name, source_branch, new_branch)
            return {"status": "branch created"}
        elif action == "create_pull_request":
            return await run_in_executor(agent.create_pull_request, repo_full_name, head_branch, base_branch, title, body)
        elif action == "add_reviewers_to_pr":
            return await run_in_executor(agent.add_reviewers_to_pr, repo_full_name, pr_number, reviewers)
        elif action == "add_pr_comment":
            return await run_in_executor(agent.add_pr_comment, repo_full_name, pr_number, body)
        elif action == "edit_pr_comment":
            return await run_in_executor(agent.edit_pr_comment, repo_full_name, comment_id, new_body)
        elif action == "list_prs_assigned_to_me":
            return {"assigned_prs": await run_in_executor(agent.list_prs_assigned_to_me, repo_full_name)}
        elif action == "list_prs_created_by_me":
            return {"my_prs": await run_in_executor(agent.list_prs_created_by_me, repo_full_name)}
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as e:
        return {"error": str(e)}

async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
