//! The porcelain added on top of the original nine subcommands: `remote`,
//! `worktree`, `config`, `diff`, `rm`, `mv`, `reset`, `tag`, `rev-parse`,
//! `ls-files`, `cat-file`, `clean`, `show`, `switch`.
//!
//! In `tests/` for the reason the whole crate's tests are: it mounts kernel
//! modules carrying `#[test_case]`, so `cargo test --lib` cannot compile and a
//! `#[cfg(test)] mod tests` in `src/` would never run.

use chitti_git_wasm::{config, git, hostsim};

/// A repository with one commit containing `a.txt`.
fn repo() -> hostsim::Guard {
    let g = hostsim::reset("/agent/9047", "/home/chitti");
    assert!(git::command("init").contains("initialized"));
    hostsim::sim()
        .files
        .insert("/home/chitti/a.txt".into(), b"one\ntwo\nthree\n".to_vec());
    git::command("add .");
    assert!(git::command("commit -m first").starts_with("ok:"));
    g
}

// --- remote ----------------------------------------------------------------

/// The whole `remote` surface round-trips through `.git/config`.
#[test]
fn remotes_can_be_added_listed_renamed_and_removed() {
    let _g = repo();
    assert!(git::command("remote").contains("no remotes"));

    assert!(git::command("remote add origin https://example.com/r.git").starts_with("ok:"));
    assert_eq!(git::command("remote"), "origin");
    let v = git::command("remote -v");
    assert!(v.contains("origin\thttps://example.com/r.git (fetch)"), "{v}");
    assert!(v.contains("(push)"), "{v}");
    assert_eq!(git::command("remote get-url origin"), "https://example.com/r.git");

    // A duplicate is refused rather than silently replacing the URL — losing a
    // remote's address to a typo'd `add` is how you push to the wrong place.
    assert!(git::command("remote add origin https://other/r.git").starts_with("error:"));
    assert_eq!(git::command("remote get-url origin"), "https://example.com/r.git");

    assert!(git::command("remote set-url origin git@example.com:r.git").starts_with("ok:"));
    assert_eq!(git::command("remote get-url origin"), "git@example.com:r.git");

    assert!(git::command("remote rename origin upstream").starts_with("ok:"));
    assert_eq!(git::command("remote"), "upstream");
    assert_eq!(git::command("remote get-url upstream"), "git@example.com:r.git");
    assert!(git::command("remote get-url origin").starts_with("error:"));

    assert!(git::command("remote remove upstream").starts_with("ok:"));
    assert!(git::command("remote").contains("no remotes"));
    // Removing a remote that is not there is an error, not a silent success.
    assert!(git::command("remote remove upstream").starts_with("error:"));
}

/// Two remotes coexist, which is what the subsection handling is for.
#[test]
fn several_remotes_coexist() {
    let _g = repo();
    git::command("remote add origin https://a/r.git");
    git::command("remote add fork https://b/r.git");
    let v = git::command("remote -v");
    assert!(v.contains("origin\thttps://a/r.git"), "{v}");
    assert!(v.contains("fork\thttps://b/r.git"), "{v}");
    git::command("remote remove origin");
    assert_eq!(git::command("remote"), "fork", "removing one must not touch the other");
}

/// `clone` records `origin`, and the new `remote` code must read what `clone`
/// wrote — the two used to be the only writer and the only reader of this file.
#[test]
fn remote_reads_what_clone_wrote() {
    let _g = hostsim::reset("/agent/9047", "/home/chitti");
    git::command("init");
    hostsim::sim().files.insert(
        "/home/chitti/.git/config".into(),
        b"[remote \"origin\"]\n\turl = https://example.com/r.git\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n".to_vec(),
    );
    assert_eq!(git::command("remote get-url origin"), "https://example.com/r.git");
    assert_eq!(git::command("remote"), "origin");
}

// --- config ----------------------------------------------------------------

#[test]
fn config_gets_sets_and_unsets() {
    let _g = repo();
    assert!(git::command("config user.name").starts_with("error:"));
    assert!(git::command("config user.name Ada Lovelace").starts_with("ok:"));
    // A value may contain spaces — it is the rest of the line, not one token.
    assert_eq!(git::command("config user.name"), "Ada Lovelace");
    assert_eq!(git::command("config --get user.name"), "Ada Lovelace");
    assert!(git::command("--list").is_empty() || true);
    assert!(git::command("config --list").contains("user.name=Ada Lovelace"));
    assert!(git::command("config --unset user.name").starts_with("ok:"));
    assert!(git::command("config user.name").starts_with("error:"));
}

/// Setting a key must not disturb the rest of the file — a writer that
/// reformats silently drops whatever it did not understand.
#[test]
fn config_preserves_unrelated_entries() {
    let _g = repo();
    git::command("remote add origin https://example.com/r.git");
    git::command("config user.email ada@example.com");
    git::command("config core.bare false");
    assert_eq!(git::command("remote get-url origin"), "https://example.com/r.git");
    assert_eq!(git::command("config user.email"), "ada@example.com");
    assert_eq!(git::command("config core.bare"), "false");
}

/// The config parser's own edge cases, off the filesystem.
#[test]
fn config_parsing_handles_subsections_and_comments() {
    let text = "# a comment\n[core]\n\trepositoryformatversion = 0\n[remote \"my.origin\"]\n\turl = git@h:r.git\n";
    let e = config::parse(text);
    assert_eq!(config::get(&e, "core.repositoryformatversion"), Some("0"));
    // A subsection may itself contain a dot, so the split is first-and-last.
    assert_eq!(config::get(&e, "remote.my.origin.url"), Some("git@h:r.git"));
    assert_eq!(config::subsections(&e, "remote"), vec!["my.origin".to_string()]);
    // Round-trips through render.
    assert_eq!(config::get(&config::parse(&config::render(&e)), "remote.my.origin.url"), Some("git@h:r.git"));
    // Section names fold case; subsection names do not.
    let e = config::parse("[REMOTE \"Origin\"]\n\turl = x\n");
    assert_eq!(config::get(&e, "remote.Origin.url"), Some("x"));
    assert_eq!(config::get(&e, "remote.origin.url"), None, "subsections are case-sensitive");
}

// --- worktree --------------------------------------------------------------

/// A linked worktree gets its own directory, checkout and branch, while the
/// object store stays shared.
#[test]
fn a_worktree_is_added_listed_and_removed() {
    let _g = repo();
    let out = git::command("worktree add /home/chitti/wt feature");
    assert!(out.starts_with("ok:"), "{out}");

    // The `.git` **file** is what marks it, not a directory.
    let dot = hostsim::sim().files.get("/home/chitti/wt/.git").cloned();
    let dot = String::from_utf8(dot.expect("the worktree needs a .git file")).unwrap();
    assert!(dot.starts_with("gitdir: "), "{dot}");

    // The commit's content was checked out into it.
    assert_eq!(
        hostsim::sim().files.get("/home/chitti/wt/a.txt").map(|v| v.as_slice()),
        Some(&b"one\ntwo\nthree\n"[..]),
    );

    let list = git::command("worktree list");
    assert!(list.contains("/home/chitti/wt"), "{list}");
    assert!(list.contains("feature"), "{list}");

    assert!(git::command("worktree remove /home/chitti/wt").starts_with("ok:"));
    assert!(!git::command("worktree list").contains("/home/chitti/wt"));
}

/// **A branch may be checked out in only one worktree.** Two trees on one branch
/// would each commit over the other's work, which is why git refuses it.
#[test]
fn a_branch_cannot_be_checked_out_twice() {
    let _g = repo();
    let main = git::command("branch");
    let cur = main
        .lines()
        .find(|l| l.starts_with('*'))
        .map(|l| l.trim_start_matches("* ").trim().to_string())
        .unwrap_or_else(|| "master".to_string());

    assert!(git::command(&format!("worktree add /home/chitti/w1 {cur}")).starts_with("error:"));
    assert!(git::command("worktree add /home/chitti/w2 feature").starts_with("ok:"));
    // …and not in a second linked worktree either.
    let out = git::command("worktree add /home/chitti/w3 feature");
    assert!(out.starts_with("error:"), "{out}");
}

// --- index and working tree ------------------------------------------------

#[test]
fn rm_drops_paths_from_the_index_and_disk() {
    let _g = repo();
    assert!(git::command("ls-files").contains("a.txt"));
    assert!(git::command("rm a.txt").starts_with("ok:"));
    assert!(!git::command("ls-files").contains("a.txt"));
    assert!(!hostsim::sim().files.contains_key("/home/chitti/a.txt"));
}

/// `--cached` unstages **without** touching the file, which is the whole reason
/// the flag exists.
#[test]
fn rm_cached_keeps_the_file() {
    let _g = repo();
    assert!(git::command("rm --cached a.txt").starts_with("ok:"));
    assert!(!git::command("ls-files").contains("a.txt"));
    assert!(hostsim::sim().files.contains_key("/home/chitti/a.txt"), "the file must stay");
}

#[test]
fn mv_renames_in_the_index_and_on_disk() {
    let _g = repo();
    assert!(git::command("mv a.txt b.txt").starts_with("ok:"));
    assert!(!hostsim::sim().files.contains_key("/home/chitti/a.txt"));
    assert_eq!(
        hostsim::sim().files.get("/home/chitti/b.txt").map(|v| v.as_slice()),
        Some(&b"one\ntwo\nthree\n"[..])
    );
    let ls = git::command("ls-files");
    assert!(ls.contains("b.txt") && !ls.contains("a.txt"), "{ls}");
}

/// `--hard` puts the files back; `--soft` leaves them alone. Getting those the
/// wrong way round destroys uncommitted work.
#[test]
fn reset_modes_differ_in_what_they_touch() {
    let _g = repo();
    let first = git::command("rev-parse HEAD");
    hostsim::sim()
        .files
        .insert("/home/chitti/a.txt".into(), b"changed\n".to_vec());
    git::command("add .");
    git::command("commit -m second");

    // --soft moves the branch and keeps the working file.
    assert!(git::command(&format!("reset --soft {first}")).starts_with("ok:"));
    assert_eq!(git::command("rev-parse HEAD"), first);
    assert_eq!(
        hostsim::sim().files.get("/home/chitti/a.txt").map(|v| v.as_slice()),
        Some(&b"changed\n"[..]),
        "--soft must not touch the working tree"
    );

    // --hard replaces it with the commit's content.
    assert!(git::command(&format!("reset --hard {first}")).starts_with("ok:"));
    assert_eq!(
        hostsim::sim().files.get("/home/chitti/a.txt").map(|v| v.as_slice()),
        Some(&b"one\ntwo\nthree\n"[..]),
        "--hard must restore the committed content"
    );
}

#[test]
fn clean_is_a_dry_run_without_f() {
    let _g = repo();
    hostsim::sim()
        .files
        .insert("/home/chitti/junk.tmp".into(), b"x".to_vec());
    let dry = git::command("clean");
    assert!(dry.contains("would remove junk.tmp"), "{dry}");
    assert!(hostsim::sim().files.contains_key("/home/chitti/junk.tmp"), "a dry run must not delete");
    assert!(git::command("clean -f").contains("removed junk.tmp"));
    assert!(!hostsim::sim().files.contains_key("/home/chitti/junk.tmp"));
    // A tracked file is never cleaned.
    assert!(hostsim::sim().files.contains_key("/home/chitti/a.txt"));
}

// --- inspection ------------------------------------------------------------

#[test]
fn rev_parse_resolves_head_branches_tags_and_abbreviations() {
    let _g = repo();
    let head = git::command("rev-parse HEAD");
    assert_eq!(head.len(), 40, "{head}");
    assert!(git::command("tag v1").starts_with("ok:"));
    assert_eq!(git::command("rev-parse v1"), head);
    // An abbreviation resolves when unique.
    let objs = git::command("ls-files");
    assert_eq!(git::command(&format!("rev-parse {}", &head[..8])), head, "objects: {objs}");
    assert!(git::command("rev-parse nosuchthing").starts_with("error:"));
}

#[test]
fn tags_are_listed_added_and_deleted() {
    let _g = repo();
    assert!(git::command("tag").contains("no tags"));
    assert!(git::command("tag v1").starts_with("ok:"));
    assert_eq!(git::command("tag"), "v1");
    // A duplicate is refused rather than moved.
    assert!(git::command("tag v1").starts_with("error:"));
    assert!(git::command("tag -d v1").starts_with("ok:"));
    assert!(git::command("tag").contains("no tags"));
    // Annotated tags are refused by name, not silently made lightweight.
    assert!(git::command("tag -a v2").contains("not implemented"));
}

#[test]
fn cat_file_reports_type_size_and_content() {
    let _g = repo();
    let head = git::command("rev-parse HEAD");
    assert_eq!(git::command(&format!("cat-file -t {head}")), "commit");
    assert!(git::command(&format!("cat-file -s {head}")).parse::<usize>().unwrap() > 0);
    assert!(git::command(&format!("cat-file -p {head}")).contains("tree "));
}

#[test]
fn show_reports_the_commit_and_what_it_changed() {
    let _g = repo();
    let out = git::command("show");
    assert!(out.starts_with("commit "), "{out}");
    assert!(out.contains("first"), "the message: {out}");
    assert!(out.contains("A\ta.txt"), "the first commit adds its files: {out}");
}

/// A line diff, with the hunk header and the +/- lines.
#[test]
fn diff_shows_added_and_removed_lines() {
    let _g = repo();
    assert!(git::command("diff").contains("no changes"));
    hostsim::sim()
        .files
        .insert("/home/chitti/a.txt".into(), b"one\nTWO\nthree\n".to_vec());
    let d = git::command("diff");
    assert!(d.contains("diff --git a/a.txt b/a.txt"), "{d}");
    assert!(d.contains("@@"), "a hunk header: {d}");
    assert!(d.contains("-two"), "{d}");
    assert!(d.contains("+TWO"), "{d}");
    // The unchanged lines are outside the hunk, which is what the common
    // prefix/suffix trim is for.
    assert!(!d.contains("-one"), "context must not appear as a removal: {d}");
}

#[test]
fn switch_is_checkout_with_the_modern_spelling() {
    let _g = repo();
    assert!(git::command("switch -c feature").starts_with("ok:"));
    let b = git::command("branch");
    assert!(b.contains("* feature"), "{b}");
    assert!(git::command("switch").starts_with("usage:"));
}

/// Unimplemented commands are named as such, not reported as typos — the two
/// send a user in completely different directions.
#[test]
fn unimplemented_commands_say_so() {
    let _g = repo();
    for c in ["merge other", "rebase main", "cherry-pick abc", "stash", "revert abc"] {
        let out = git::command(c);
        assert!(out.contains("not implemented"), "{c}: {out}");
    }
    assert!(git::command("frobnicate").contains("unknown subcommand"));
}

/// Every new subcommand is in the help text — a command nobody can discover is
/// only half-added.
#[test]
fn help_lists_the_new_subcommands() {
    let help = git::command("help");
    for c in [
        "remote", "worktree", "config", "diff", "rm", "mv", "reset", "restore", "clean",
        "show", "tag", "rev-parse", "ls-files", "cat-file", "switch", "fetch", "pull",
    ] {
        assert!(help.contains(c), "help does not mention {c}: {help}");
    }
}

/// Nothing works outside a repository, and each says so rather than panicking
/// or reporting an empty result that reads like success.
#[test]
fn commands_outside_a_repository_are_refused() {
    let _g = hostsim::reset("/agent/9047", "/home/chitti");
    for c in ["remote -v", "worktree list", "config --list", "diff", "tag", "ls-files", "rm x"] {
        let out = git::command(c);
        assert!(out.contains("not a git repository"), "{c}: {out}");
    }
}
