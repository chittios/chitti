//! **Cooperative web workers & load jobs** for the browser agent.
//!
//! Reference (not linked):
//! - Ladybird `Libraries/LibWeb/HTML/Worker.{h,cpp}` — dedicated Worker interface
//! - Ladybird `Libraries/LibWeb/HTML/WorkerGlobalScope.*` — worker global
//! - Ladybird `Libraries/LibWeb/HTML/Scripting/WorkerAgent.h` — agent isolation
//! - Ladybird `Services/WebWorker/` — out-of-process worker host
//! - HTML living standard: dedicated workers + `postMessage`
//!
//! Chitti's shell path is **cooperative single-threaded** (determinism
//! boundary + scheduler). We cannot spawn OS threads for each Worker. Instead:
//!
//! 1. **Load jobs** — queue of resource fetches (images/scripts/styles) that
//!    the page needs; the shell drains them with upkeep + Ctrl+C between jobs
//!    (Ladybird's async ResourceLoader completions, serialized).
//! 2. **Script workers** — isolated JS runs (`js::run_scripts` on a private
//!    DOM stub) with an inbox/outbox message queue (`postMessage` / `onmessage`
//!    subset). Scripts are already loaded bytes (via the loader) or inline.
//!
//! No ambient authority: network only through [`super::loader`], which is
//! capability-gated at the shell/tool layer.

use super::cache::{AssetStore, CacheMode};
use super::loader::{self, Destination, LoadRequest, LoadedResource};
use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub type JobId = u64;

/// Work item for the cooperative pool.
#[derive(Clone, Debug)]
pub enum Job {
    /// Fetch a URL through the resource loader + optional asset store.
    Fetch {
        id: JobId,
        url: String,
        destination: Destination,
        cache_mode: CacheMode,
        timeout_ms: u64,
    },
    /// Run a worker script source in isolation; messages land in outbox.
    Script {
        id: JobId,
        name: String,
        source: String,
        /// Initial messages already posted from the page (`postMessage` before run).
        inbox: Vec<String>,
    },
}

/// Outcome of a finished job.
#[derive(Clone, Debug)]
pub enum JobResult {
    FetchOk {
        id: JobId,
        resource: LoadedResource,
    },
    FetchErr {
        id: JobId,
        url: String,
        error: String,
    },
    ScriptDone {
        id: JobId,
        name: String,
        /// `console.log` lines from the worker script.
        log: Vec<String>,
        /// Messages the worker "posted" (our subset: `self.postMessage` not
        /// yet in js.rs — we surface console + a synthetic done message).
        messages: Vec<String>,
    },
}

/// Dedicated-worker-like handle (Ladybird `HTML::Worker` outside port).
#[derive(Clone, Debug)]
pub struct WorkerHandle {
    pub id: JobId,
    pub name: String,
    pub script_url: String,
    /// Messages from the page toward the worker (filled before/during run).
    pub pending_in: Vec<String>,
    /// Messages the worker produced.
    pub outbox: Vec<String>,
    pub terminated: bool,
}

/// Cooperative job queue + worker registry.
pub struct WorkerPool {
    next_id: JobId,
    queue: VecDeque<Job>,
    results: Vec<JobResult>,
    workers: Vec<WorkerHandle>,
    /// Shared page asset store (URL → bytes) filled by fetch jobs.
    pub assets: AssetStore,
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerPool {
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            queue: VecDeque::new(),
            results: Vec::new(),
            workers: Vec::new(),
            assets: AssetStore {
                map: BTreeMap::new(),
            },
        }
    }

    pub fn assets(&self) -> &AssetStore {
        &self.assets
    }

    pub fn assets_mut(&mut self) -> &mut AssetStore {
        &mut self.assets
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    pub fn worker_count(&self) -> usize {
        self.workers.iter().filter(|w| !w.terminated).count()
    }

    fn alloc_id(&mut self) -> JobId {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// Enqueue a subresource fetch (image/script/style/…).
    pub fn spawn_fetch(
        &mut self,
        url: &str,
        destination: Destination,
        cache_mode: CacheMode,
        timeout_ms: u64,
    ) -> JobId {
        let id = self.alloc_id();
        self.queue.push_back(Job::Fetch {
            id,
            url: url.to_string(),
            destination,
            cache_mode,
            timeout_ms,
        });
        id
    }

    /// Enqueue image fetches for every non-empty src (absolute URLs expected).
    pub fn spawn_image_fetches(&mut self, urls: &[String]) -> Vec<JobId> {
        urls.iter()
            .filter(|u| !u.is_empty())
            .map(|u| self.spawn_fetch(u, Destination::Image, CacheMode::Default, 20_000))
            .collect()
    }

    /// Create a dedicated worker from inline source (Ladybird Worker::create
    /// after script URL resolve — we skip the separate fetch when source is known).
    pub fn spawn_script_worker(&mut self, name: &str, source: &str) -> JobId {
        let id = self.alloc_id();
        self.workers.push(WorkerHandle {
            id,
            name: name.to_string(),
            script_url: String::from("(inline)"),
            pending_in: Vec::new(),
            outbox: Vec::new(),
            terminated: false,
        });
        self.queue.push_back(Job::Script {
            id,
            name: name.to_string(),
            source: source.to_string(),
            inbox: Vec::new(),
        });
        id
    }

    /// Create a worker that first fetches `script_url` then runs it.
    pub fn spawn_worker_url(&mut self, name: &str, script_url: &str) -> JobId {
        let id = self.spawn_fetch(
            script_url,
            Destination::Worker,
            CacheMode::Default,
            30_000,
        );
        self.workers.push(WorkerHandle {
            id,
            name: name.to_string(),
            script_url: script_url.to_string(),
            pending_in: Vec::new(),
            outbox: Vec::new(),
            terminated: false,
        });
        id
    }

    /// Page → worker message (queued until the Script job runs).
    pub fn post_message(&mut self, worker_id: JobId, msg: &str) -> bool {
        if let Some(w) = self.workers.iter_mut().find(|w| w.id == worker_id) {
            if w.terminated {
                return false;
            }
            w.pending_in.push(msg.to_string());
            // Also attach to a still-queued Script job if present.
            for job in self.queue.iter_mut() {
                if let Job::Script { id, inbox, .. } = job {
                    if *id == worker_id {
                        inbox.push(msg.to_string());
                    }
                }
            }
            return true;
        }
        false
    }

    /// Terminate a worker (Ladybird `Worker::terminate`) — drops pending script jobs.
    pub fn terminate(&mut self, worker_id: JobId) -> bool {
        let mut found = false;
        for w in self.workers.iter_mut() {
            if w.id == worker_id {
                w.terminated = true;
                found = true;
            }
        }
        self.queue.retain(|j| match j {
            Job::Script { id, .. } => *id != worker_id,
            _ => true,
        });
        found
    }

    /// Run **one** job to completion. Returns `true` if a job ran.
    /// Network jobs pump upkeep inside `loader::load` → `get_follow`.
    pub fn poll_one(&mut self) -> bool {
        let Some(job) = self.queue.pop_front() else {
            return false;
        };
        match job {
            Job::Fetch {
                id,
                url,
                destination,
                cache_mode,
                timeout_ms,
            } => {
                let req = LoadRequest {
                    url: url.clone(),
                    method: String::from("GET"),
                    cache_mode,
                    destination,
                    timeout_ms,
                    source_url: None,
                    cors_mode: crate::browser::cors::RequestMode::NoCors,
                    initiator_origin: None,
                    credentials: false,
                };
                match loader::load_into_assets(&mut self.assets, &req) {
                    Ok(resource) => {
                        // If this was a worker script fetch, queue a Script job.
                        if destination == Destination::Worker {
                            if let Some(w) = self.workers.iter().find(|w| w.id == id) {
                                if !w.terminated {
                                    let name = w.name.clone();
                                    let source = resource.text();
                                    let inbox = w.pending_in.clone();
                                    self.queue.push_back(Job::Script {
                                        id,
                                        name,
                                        source,
                                        inbox,
                                    });
                                }
                            }
                        }
                        self.results.push(JobResult::FetchOk { id, resource });
                    }
                    Err(error) => {
                        self.results.push(JobResult::FetchErr { id, url, error });
                    }
                }
            }
            Job::Script {
                id,
                name,
                source,
                inbox,
            } => {
                if self
                    .workers
                    .iter()
                    .any(|w| w.id == id && w.terminated)
                {
                    return true;
                }
                let (log, messages) = run_worker_script(&name, &source, &inbox);
                if let Some(w) = self.workers.iter_mut().find(|w| w.id == id) {
                    w.outbox.extend(messages.iter().cloned());
                }
                self.results.push(JobResult::ScriptDone {
                    id,
                    name,
                    log,
                    messages,
                });
            }
        }
        true
    }

    /// Drain all jobs until empty or `should_stop` returns true (Ctrl+C).
    /// Returns number of jobs completed.
    pub fn drain_while(&mut self, mut should_stop: impl FnMut() -> bool) -> usize {
        let mut n = 0;
        while !self.queue.is_empty() {
            if should_stop() {
                break;
            }
            // UI pump between jobs (same rule as shell long ops).
            #[cfg(not(test))]
            {
                crate::shell::upkeep();
            }
            if self.poll_one() {
                n += 1;
            } else {
                break;
            }
        }
        n
    }

    /// Take finished results (caller owns them).
    pub fn take_results(&mut self) -> Vec<JobResult> {
        core::mem::take(&mut self.results)
    }

    /// Reset pool state (new navigation).
    pub fn reset(&mut self) {
        self.queue.clear();
        self.results.clear();
        self.workers.clear();
        self.assets.clear();
    }
}

/// Run worker source with an isolated mini-DOM. Inbox messages are exposed as
/// `self.name` only for now; we append a done sentinel. Console log is captured
/// via the existing JS engine.
fn run_worker_script(name: &str, source: &str, inbox: &[String]) -> (Vec<String>, Vec<String>) {
    // Prefix a tiny worker shim so scripts can read a name and inbox length.
    // Full `onmessage` / `postMessage` needs richer js.rs host objects later.
    let mut wrapped = String::new();
    wrapped.push_str("// worker: ");
    wrapped.push_str(name);
    wrapped.push('\n');
    wrapped.push_str(&format!(
        "var __workerName = \"{}\";\n",
        escape_js_string(name)
    ));
    wrapped.push_str(&format!("var __inboxLen = {};\n", inbox.len()));
    if let Some(first) = inbox.first() {
        wrapped.push_str(&format!(
            "var __firstMessage = \"{}\";\n",
            escape_js_string(first)
        ));
    }
    wrapped.push_str(source);

    // Minimal HTML shell so js::run_scripts has a document.
    let html = format!(
        "<html><head><title>worker:{name}</title><script>{wrapped}</script></head><body></body></html>"
    );
    let doc = super::html::parse(&html);
    let mut dom = super::js::JsDom::from_document(&doc);
    let scripts = doc.scripts.clone();
    let log = super::js::run_scripts(&mut dom, &scripts);

    let mut messages = Vec::new();
    for m in inbox {
        messages.push(format!("echo:{m}"));
    }
    messages.push(String::from("worker-done"));
    (log, messages)
}

fn escape_js_string(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            _ => o.push(c),
        }
    }
    o
}

/// Process-wide registry for long-lived dedicated workers (not held across
/// network I/O — `mm::Locked` disables interrupts for the critical section).
pub static POOL: crate::mm::Locked<WorkerPool> = crate::mm::Locked::new(WorkerPool::new());

/// Spawn image loads on a **stack-local** pool and drain cooperatively.
/// Must not hold `POOL` across `loader::load` (interrupts-off spinlock).
/// Returns successfully loaded image resources plus the session asset store.
pub fn fetch_images_cooperative(urls: &[String]) -> (Vec<LoadedResource>, AssetStore) {
    let mut pool = WorkerPool::new();
    pool.spawn_image_fetches(urls);
    pool.drain_while(|| {
        #[cfg(not(test))]
        {
            crate::shell::poll_interrupt()
        }
        #[cfg(test)]
        {
            false
        }
    });
    let results = pool.take_results();
    let loaded: Vec<LoadedResource> = results
        .into_iter()
        .filter_map(|r| match r {
            JobResult::FetchOk { resource, .. } if resource.status < 400 => Some(resource),
            _ => None,
        })
        .collect();
    let assets = core::mem::take(&mut pool.assets);
    (loaded, assets)
}

/// Most subresource URLs fetched per [`fetch_subresources_cooperative`] call.
const MAX_SUBRESOURCE_URLS: usize = 16;

/// Fetch a batch of same-kind subresources (scripts / styles / fonts) on a
/// **stack-local** pool and drain cooperatively (upkeep + Ctrl+C between
/// jobs, same as [`fetch_images_cooperative`]). Returns requested-URL → body
/// for responses with status < 400 whose body is at most `per_item_cap`
/// bytes; failed or oversized fetches record nothing. At most 16 URLs are
/// fetched (extras ignored).
pub fn fetch_subresources_cooperative(
    urls: &[String],
    dest: Destination,
    per_item_cap: usize,
) -> BTreeMap<String, Vec<u8>> {
    let mut pool = WorkerPool::new();
    fetch_subresources_into(&mut pool, urls, dest, per_item_cap)
}

/// Body of [`fetch_subresources_cooperative`] on a caller-owned pool
/// (testable with a pre-seeded asset store, no network).
fn fetch_subresources_into(
    pool: &mut WorkerPool,
    urls: &[String],
    dest: Destination,
    per_item_cap: usize,
) -> BTreeMap<String, Vec<u8>> {
    // Per-kind timeouts: styles block layout (shorter), scripts/fonts get 30 s.
    let timeout_ms = match dest {
        Destination::Style => 20_000,
        _ => 30_000,
    };
    let mut ids: BTreeMap<JobId, String> = BTreeMap::new();
    for url in urls
        .iter()
        .filter(|u| !u.is_empty())
        .take(MAX_SUBRESOURCE_URLS)
    {
        let id = pool.spawn_fetch(url, dest, CacheMode::Default, timeout_ms);
        ids.insert(id, url.clone());
    }
    pool.drain_while(|| {
        #[cfg(not(test))]
        {
            crate::shell::poll_interrupt()
        }
        #[cfg(test)]
        {
            false
        }
    });
    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for r in pool.take_results() {
        if let JobResult::FetchOk { id, resource } = r {
            if resource.status < 400 && resource.body.len() <= per_item_cap {
                if let Some(url) = ids.get(&id) {
                    out.insert(url.clone(), resource.body);
                }
            }
        }
    }
    out
}

/// Clear global worker registry (new top-level navigation).
pub fn reset_global() {
    POOL.with(|p| p.reset());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn script_worker_runs_and_logs() {
        let mut pool = WorkerPool::new();
        let id = pool.spawn_script_worker(
            "w1",
            r#"console.log("hello-worker"); document.title = "W";"#,
        );
        assert_eq!(pool.pending(), 1);
        assert!(pool.poll_one());
        let results = pool.take_results();
        assert_eq!(results.len(), 1);
        match &results[0] {
            JobResult::ScriptDone {
                id: rid,
                log,
                messages,
                ..
            } => {
                assert_eq!(*rid, id);
                assert!(
                    log.iter().any(|l| l.contains("hello-worker")),
                    "log={log:?}"
                );
                assert!(messages.iter().any(|m| m == "worker-done"));
            }
            other => panic!("expected ScriptDone, got {other:?}"),
        }
        assert_eq!(pool.worker_count(), 1);
    }

    #[test_case]
    fn post_message_and_terminate() {
        let mut pool = WorkerPool::new();
        let id = pool.spawn_script_worker("w", "console.log(__inboxLen);");
        assert!(pool.post_message(id, "ping"));
        // Message attached to queued job.
        pool.terminate(id);
        assert_eq!(pool.pending(), 0, "terminate drops script job");
        assert_eq!(pool.worker_count(), 0);
    }

    #[test_case]
    fn fetch_job_uses_asset_store_without_network_on_second_hit() {
        let mut pool = WorkerPool::new();
        pool.assets_mut()
            .put("https://ex.com/a.png", "image/png", b"\x89PNG\r\n".to_vec());
        let id = pool.spawn_fetch(
            "https://ex.com/a.png",
            Destination::Image,
            CacheMode::Default,
            1000,
        );
        assert!(pool.poll_one());
        let results = pool.take_results();
        match &results[0] {
            JobResult::FetchOk { id: rid, resource } => {
                assert_eq!(*rid, id);
                assert!(resource.from_cache);
                assert_eq!(resource.body, b"\x89PNG\r\n");
            }
            other => panic!("expected FetchOk, got {other:?}"),
        }
    }

    #[test_case]
    fn spawn_image_fetches_filters_empty() {
        let mut pool = WorkerPool::new();
        let urls = alloc::vec![
            String::from("https://a/x.png"),
            String::new(),
            String::from("https://b/y.png"),
        ];
        let ids = pool.spawn_image_fetches(&urls);
        assert_eq!(ids.len(), 2);
        assert_eq!(pool.pending(), 2);
    }

    #[test_case]
    fn escape_js_string_basic() {
        assert_eq!(escape_js_string("a\"b\\c\n"), "a\\\"b\\\\c\\n");
    }

    #[test_case]
    fn fetch_subresources_collects_and_caps() {
        let mut pool = WorkerPool::new();
        pool.assets_mut()
            .put("https://ex.com/a.css", "text/css", b"p{color:red}".to_vec());
        pool.assets_mut().put(
            "https://ex.com/big.css",
            "text/css",
            alloc::vec![b'x'; 64],
        );
        let urls = alloc::vec![
            String::from("https://ex.com/a.css"),
            String::new(),
            String::from("https://ex.com/big.css"),
        ];
        let out = fetch_subresources_into(&mut pool, &urls, Destination::Style, 32);
        assert_eq!(out.len(), 1, "oversized + empty skipped: {out:?}");
        assert_eq!(
            out.get("https://ex.com/a.css").map(|b| b.as_slice()),
            Some(b"p{color:red}".as_slice())
        );
        assert!(!out.contains_key("https://ex.com/big.css"));
    }

    #[test_case]
    fn fetch_subresources_truncates_to_sixteen() {
        let mut pool = WorkerPool::new();
        let mut urls = Vec::new();
        for i in 0..20 {
            let u = format!("https://ex.com/{i}.js");
            pool.assets_mut()
                .put(&u, "text/javascript", alloc::vec![b'a']);
            urls.push(u);
        }
        let out = fetch_subresources_into(&mut pool, &urls, Destination::Script, 1024);
        assert_eq!(out.len(), MAX_SUBRESOURCE_URLS);
    }

    #[test_case]
    fn drain_while_stops_on_predicate() {
        let mut pool = WorkerPool::new();
        pool.spawn_script_worker("a", "console.log(1);");
        pool.spawn_script_worker("b", "console.log(2);");
        let mut n = 0;
        let done = pool.drain_while(|| {
            n += 1;
            n > 1
        });
        // First job runs (n becomes 1, stop is false), second iteration n=2 stop.
        assert!(done >= 1);
        assert!(pool.pending() <= 1);
    }
}
