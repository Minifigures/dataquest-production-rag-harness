"""The GitQuest corpus: a small set of git documentation chunks.

Kept in the repository as plain text rather than fetched, so retrieval,
grounding and the whole harness are deterministic and run with no network and
no API key. Each chunk has a stable id, which is what citations refer to and
what the faithfulness check scores against.

The `topic` field is the slice dimension the monitoring code groups by.
"""

CHUNKS = [
    {"chunk_id": "git-branch-01", "topic": "branching",
     "text": "A branch in git is a lightweight movable pointer to a commit. "
             "Creating a branch with git branch <name> does not copy any "
             "files; it writes a single file under .git/refs/heads containing "
             "the commit hash. This is why branching in git is nearly "
             "instantaneous regardless of repository size."},
    {"chunk_id": "git-branch-02", "topic": "branching",
     "text": "git checkout -b <name> creates a branch and switches to it in "
             "one step. Since git 2.23 the preferred command is git switch -c "
             "<name>, which was introduced to separate branch switching from "
             "file restoration, two unrelated jobs that checkout had both."},
    {"chunk_id": "git-merge-01", "topic": "merging",
     "text": "A fast-forward merge happens when the target branch has not "
             "diverged from the source. Git simply moves the branch pointer "
             "forward and no merge commit is created. Passing --no-ff forces "
             "a merge commit even when a fast-forward is possible, which "
             "keeps the fact that a branch existed visible in history."},
    {"chunk_id": "git-merge-02", "topic": "merging",
     "text": "A three-way merge is used when both branches have new commits. "
             "Git finds the merge base, the most recent common ancestor, and "
             "combines the two sets of changes relative to it. Conflicts are "
             "reported only where both sides changed the same region."},
    {"chunk_id": "git-rebase-01", "topic": "rebasing",
     "text": "git rebase replays commits from one branch onto another, "
             "creating new commits with new hashes. The original commits are "
             "not modified but are left unreferenced. Because the hashes "
             "change, rebasing commits that others have already pulled "
             "rewrites shared history and should be avoided."},
    {"chunk_id": "git-rebase-02", "topic": "rebasing",
     "text": "Interactive rebase, git rebase -i, opens a list of commits that "
             "can be reordered, squashed, edited or dropped before they are "
             "replayed. It is the usual way to tidy a feature branch before "
             "review."},
    {"chunk_id": "git-reset-01", "topic": "undoing",
     "text": "git reset --soft moves the branch pointer and leaves both the "
             "index and the working tree untouched. git reset --mixed, the "
             "default, also resets the index. git reset --hard resets the "
             "working tree as well, discarding uncommitted changes "
             "permanently."},
    {"chunk_id": "git-revert-01", "topic": "undoing",
     "text": "git revert creates a new commit that undoes the changes of an "
             "earlier commit, leaving history intact. It is the safe way to "
             "undo a change that has already been pushed, because it adds to "
             "history rather than rewriting it."},
    {"chunk_id": "git-stash-01", "topic": "stashing",
     "text": "git stash saves uncommitted changes onto a stack and restores a "
             "clean working tree. git stash pop reapplies the most recent "
             "entry and removes it from the stack, while git stash apply "
             "reapplies it and keeps it. Untracked files are not stashed "
             "unless -u is passed."},
    {"chunk_id": "git-remote-01", "topic": "remotes",
     "text": "git fetch downloads objects and refs from a remote without "
             "changing the working tree. git pull is fetch followed by a "
             "merge of the upstream branch into the current one. Setting "
             "pull.rebase true makes git pull rebase instead of merging."},
    {"chunk_id": "git-remote-02", "topic": "remotes",
     "text": "A force push with --force overwrites the remote branch and can "
             "discard commits other people have pushed. --force-with-lease "
             "refuses the push if the remote has moved since you last "
             "fetched, which makes it the safer option."},
    {"chunk_id": "git-tag-01", "topic": "tagging",
     "text": "A lightweight tag is just a named pointer to a commit. An "
             "annotated tag, created with git tag -a, is a full object "
             "storing a tagger, a date and a message, and can be signed. "
             "Releases should use annotated tags."},
    {"chunk_id": "git-log-01", "topic": "history",
     "text": "git log --oneline --graph --decorate shows a compact commit "
             "graph with branch and tag names. git log -- <path> limits the "
             "log to commits touching a path, and git log -S<string> finds "
             "commits that changed the number of occurrences of a string."},
    {"chunk_id": "git-bisect-01", "topic": "history",
     "text": "git bisect performs a binary search across commits to find the "
             "one that introduced a bug. After git bisect start, mark a known "
             "bad commit and a known good commit, and git checks out the "
             "midpoint for you to test, halving the range each round."},
    {"chunk_id": "git-config-01", "topic": "configuration",
     "text": "Git reads configuration from three levels: system, global for "
             "the user, and local to the repository. The most specific level "
             "wins, so a repository-local setting overrides the global one. "
             "git config --list --show-origin shows where each value came "
             "from."},
    {"chunk_id": "git-hooks-01", "topic": "configuration",
     "text": "Hooks are scripts in .git/hooks that run at points in the git "
             "lifecycle, such as pre-commit and pre-push. They are not "
             "copied when a repository is cloned, which is why teams use a "
             "tool to install them rather than relying on the directory."},
]

BY_ID = {chunk["chunk_id"]: chunk for chunk in CHUNKS}
TOPICS = sorted({chunk["topic"] for chunk in CHUNKS})


def get(chunk_id):
    return BY_ID.get(chunk_id)


def texts_for(chunk_ids):
    return [BY_ID[c]["text"] for c in chunk_ids if c in BY_ID]
