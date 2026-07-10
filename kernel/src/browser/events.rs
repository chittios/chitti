//! Browser **event loop** — tasks, microtasks, and listeners.
//!
//! Reference: HTML event loop, Ladybird `Libraries/LibWeb/HTML/EventLoop/`,
//! DOM `EventTarget` / `addEventListener` / `dispatchEvent`.
//!
//! Cooperative: the shell drains queues between UI pumps (no OS threads).
//! This is a complete *structure* for the subset event model (no full UI
//! events for every mouse pixel — click/keydown/message/load/storage).

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub type ListenerId = u64;
pub type TargetId = u64; // 0 = window, 1 = document, 2+ = element indices

#[derive(Clone, Debug)]
pub struct Event {
    pub type_: String,
    pub target: TargetId,
    pub bubbles: bool,
    pub cancelable: bool,
    pub data: String,
    pub origin: String,
    pub default_prevented: bool,
    pub stopped: bool,
}

impl Event {
    pub fn new(type_: &str, target: TargetId) -> Self {
        Self {
            type_: type_.to_string(),
            target,
            bubbles: true,
            cancelable: true,
            data: String::new(),
            origin: String::new(),
            default_prevented: false,
            stopped: false,
        }
    }

    pub fn with_data(mut self, data: &str) -> Self {
        self.data = data.to_string();
        self
    }

    pub fn prevent_default(&mut self) {
        if self.cancelable {
            self.default_prevented = true;
        }
    }

    pub fn stop_propagation(&mut self) {
        self.stopped = true;
    }
}

#[derive(Clone, Debug)]
pub struct Listener {
    pub id: ListenerId,
    pub target: TargetId,
    pub type_: String,
    pub once: bool,
    pub capture: bool,
    /// Script body or host callback name (`"console"` dumps to log).
    pub handler: String,
}

#[derive(Clone, Debug)]
pub enum Task {
    /// Fire a DOM event through the listener path.
    Dispatch(Event),
    /// Run a script snippet (timer / postMessage delivery).
    Script { source: String },
    /// Invoke a named host callback.
    Host { name: String, arg: String },
}

/// HTML event loop subset: task queue + microtask queue + listeners.
#[derive(Clone, Debug, Default)]
pub struct EventLoop {
    pub tasks: VecDeque<Task>,
    pub microtasks: VecDeque<Task>,
    listeners: Vec<Listener>,
    next_listener: ListenerId,
    /// Collected log from host handlers during drain (tests / diagnostics).
    pub log: Vec<String>,
    /// Last prevented-default event type.
    pub last_prevented: Option<String>,
}

impl EventLoop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_event_listener(
        &mut self,
        target: TargetId,
        type_: &str,
        handler: &str,
        once: bool,
        capture: bool,
    ) -> ListenerId {
        let id = self.next_listener;
        self.next_listener = self.next_listener.saturating_add(1);
        self.listeners.push(Listener {
            id,
            target,
            type_: type_.to_string(),
            once,
            capture,
            handler: handler.to_string(),
        });
        id
    }

    pub fn remove_event_listener(&mut self, id: ListenerId) -> bool {
        let n = self.listeners.len();
        self.listeners.retain(|l| l.id != id);
        self.listeners.len() != n
    }

    pub fn queue_task(&mut self, task: Task) {
        self.tasks.push_back(task);
    }

    pub fn queue_microtask(&mut self, task: Task) {
        self.microtasks.push_back(task);
    }

    /// Queue a DOM event for later dispatch.
    pub fn queue_event(&mut self, event: Event) {
        self.tasks.push_back(Task::Dispatch(event));
    }

        /// Build the propagation path: window → document → … → target
    /// (Ladybird EventDispatcher path construction, simplified indices).
    pub fn propagation_path(target: TargetId) -> Vec<TargetId> {
        // Targets: 0=window, 1=document, 2+=elements. Path is window, document,
        // then synthetic ancestors (even ids) down to target.
        let mut path = alloc::vec![TARGET_WINDOW, TARGET_DOCUMENT];
        if target > TARGET_DOCUMENT {
            // Ancestors every step of 1 from 2..=target (flat tree approximation).
            for t in TARGET_ELEMENT0..=target {
                if !path.contains(&t) {
                    path.push(t);
                }
            }
        }
        path
    }

    /// Synchronously dispatch (HTML DOM dispatch: capture → target → bubble).
    pub fn dispatch(&mut self, mut event: Event) -> bool {
        let path = Self::propagation_path(event.target);
        let mut to_remove = Vec::new();
        let listeners: Vec<_> = self.listeners.clone();

        // CAPTURING_PHASE: window → … → parent of target
        for &node in &path {
            if node == event.target {
                break;
            }
            for l in listeners
                .iter()
                .filter(|l| l.capture && l.type_ == event.type_ && l.target == node)
            {
                self.invoke_phase(&l.handler, &event, "capture");
                if l.once {
                    to_remove.push(l.id);
                }
                if event.stopped {
                    break;
                }
            }
            if event.stopped {
                break;
            }
        }

        // AT_PHASE: listeners on target (capture flag ignored — both fire)
        if !event.stopped {
            for l in listeners
                .iter()
                .filter(|l| l.type_ == event.type_ && l.target == event.target)
            {
                self.invoke_phase(&l.handler, &event, "target");
                if l.once {
                    to_remove.push(l.id);
                }
                if event.stopped {
                    break;
                }
            }
        }

        // BUBBLING_PHASE: target's parent → … → window
        if !event.stopped && event.bubbles {
            for &node in path.iter().rev() {
                if node == event.target {
                    continue;
                }
                for l in listeners
                    .iter()
                    .filter(|l| !l.capture && l.type_ == event.type_ && l.target == node)
                {
                    self.invoke_phase(&l.handler, &event, "bubble");
                    if l.once {
                        to_remove.push(l.id);
                    }
                    if event.stopped {
                        break;
                    }
                }
                if event.stopped {
                    break;
                }
            }
        }

        for id in to_remove {
            self.remove_event_listener(id);
        }
        if event.default_prevented {
            self.last_prevented = Some(event.type_.clone());
        }
        !event.default_prevented
    }

    /// Queue UI events from the shell (mouse/key).
    pub fn queue_ui_click(&mut self, target: TargetId, x: i32, y: i32) {
        let mut e = Event::new("click", target);
        e.data = alloc::format!("{x},{y}");
        self.queue_event(e);
        self.queue_event(Event::new("mousedown", target).with_data(&alloc::format!("{x},{y}")));
        self.queue_event(Event::new("mouseup", target).with_data(&alloc::format!("{x},{y}")));
    }

    pub fn queue_ui_keydown(&mut self, target: TargetId, key: &str) {
        self.queue_event(Event::new("keydown", target).with_data(key));
        self.queue_event(Event::new("keypress", target).with_data(key));
        self.queue_event(Event::new("keyup", target).with_data(key));
    }

    pub fn queue_load(&mut self) {
        self.queue_event(Event::new("DOMContentLoaded", TARGET_DOCUMENT));
        self.queue_event(Event::new("load", TARGET_WINDOW));
    }

    fn invoke(&mut self, handler: &str, event: &Event) {
        self.invoke_phase(handler, event, "target");
    }

    fn invoke_phase(&mut self, handler: &str, event: &Event, phase: &str) {
        if handler == "console" || handler.starts_with("log:") {
            let msg = if let Some(rest) = handler.strip_prefix("log:") {
                rest.to_string()
            } else {
                format!("event:{} phase={} data={}", event.type_, phase, event.data)
            };
            self.log.push(msg);
            return;
        }
        self.log.push(format!(
            "handler:{} type={} phase={}",
            handler, event.type_, phase
        ));
    }

    /// Run microtasks until empty, then one task (HTML perform a microtask checkpoint).
    pub fn turn(&mut self) -> bool {
        while let Some(t) = self.microtasks.pop_front() {
            self.run_task(t);
        }
        if let Some(t) = self.tasks.pop_front() {
            self.run_task(t);
            // Checkpoint after each task.
            while let Some(m) = self.microtasks.pop_front() {
                self.run_task(m);
            }
            return true;
        }
        false
    }

    /// Drain up to `budget` tasks (cooperative).
    pub fn drain(&mut self, budget: usize) -> usize {
        let mut n = 0;
        while n < budget {
            if !self.turn() {
                break;
            }
            n += 1;
        }
        n
    }

    fn run_task(&mut self, task: Task) {
        match task {
            Task::Dispatch(ev) => {
                let _ = self.dispatch(ev);
            }
            Task::Script { source } => {
                self.log.push(format!("script:{}", source.chars().take(64).collect::<String>()));
            }
            Task::Host { name, arg } => {
                self.log.push(format!("host:{name}:{arg}"));
            }
        }
    }

    pub fn pending(&self) -> usize {
        self.tasks.len() + self.microtasks.len()
    }
}

/// Window-level constants for target ids.
pub const TARGET_WINDOW: TargetId = 0;
pub const TARGET_DOCUMENT: TargetId = 1;
pub const TARGET_ELEMENT0: TargetId = 2;

/// Process-wide event loop for the browser page (drained by shell upkeep).
pub static EVENT_LOOP: crate::mm::Locked<EventLoop> = crate::mm::Locked::new(EventLoop {
    tasks: VecDeque::new(),
    microtasks: VecDeque::new(),
    listeners: Vec::new(),
    next_listener: 0,
    log: Vec::new(),
    last_prevented: None,
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn listener_dispatch_and_once() {
        let mut el = EventLoop::new();
        let id = el.add_event_listener(TARGET_WINDOW, "click", "console", true, false);
        assert_eq!(id, 0);
        el.queue_event(Event::new("click", TARGET_WINDOW).with_data("x"));
        assert_eq!(el.drain(4), 1);
        assert!(el.log.iter().any(|l| l.contains("click")));
        // once: listener removed
        el.queue_event(Event::new("click", TARGET_WINDOW));
        el.drain(4);
        assert_eq!(el.listeners.len(), 0);
    }

    #[test_case]
    fn microtask_before_next_task() {
        let mut el = EventLoop::new();
        el.queue_task(Task::Host {
            name: "a".into(),
            arg: String::new(),
        });
        el.queue_microtask(Task::Host {
            name: "micro".into(),
            arg: String::new(),
        });
        el.turn();
        // Micro runs first in turn(), then task.
        assert!(el.log[0].contains("micro"), "{:?}", el.log);
        assert!(el.log[1].contains("host:a"), "{:?}", el.log);
    }

    #[test_case]
    fn prevent_default_recorded() {
        let mut el = EventLoop::new();
        let mut ev = Event::new("submit", TARGET_DOCUMENT);
        ev.prevent_default();
        assert!(ev.default_prevented);
        let _ = el.dispatch(ev);
        assert_eq!(el.last_prevented.as_deref(), Some("submit"));
    }

    #[test_case]
    fn capture_then_bubble_order() {
        let mut el = EventLoop::new();
        el.add_event_listener(TARGET_WINDOW, "click", "log:win-cap", false, true);
        el.add_event_listener(TARGET_WINDOW, "click", "log:win-bub", false, false);
        el.add_event_listener(TARGET_ELEMENT0, "click", "log:el", false, false);
        let _ = el.dispatch(Event::new("click", TARGET_ELEMENT0));
        let joined = el.log.join("|");
        let cap = joined.find("win-cap").unwrap();
        let elp = joined.find("log:el").or_else(|| joined.find("el")).unwrap();
        let bub = joined.find("win-bub").unwrap();
        assert!(cap < elp && elp < bub, "order: {joined}");
    }
}
