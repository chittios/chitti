//! Heap management for the JavaScript runtime.
//!
//! This module provides heap allocation tracking with optional memory limits.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;

/// Configuration for the heap manager.
#[derive(Debug, Clone)]
pub struct HeapConfig {
    /// Maximum heap size in bytes. None means unlimited.
    pub max_bytes: Option<usize>,
}

impl HeapConfig {
    /// Create a new heap configuration with no memory limit.
    pub fn unlimited() -> Self {
        HeapConfig { max_bytes: None }
    }

    /// Create a new heap configuration with a memory limit.
    pub fn with_limit(max_bytes: usize) -> Self {
        HeapConfig {
            max_bytes: Some(max_bytes),
        }
    }
}

impl Default for HeapConfig {
    fn default() -> Self {
        Self::unlimited()
    }
}

/// Heap manager for tracking memory allocations.
#[derive(Debug)]
pub struct Heap {
    config: HeapConfig,
    allocated_bytes: usize,
}

impl Heap {
    /// Create a new heap with the given configuration.
    pub fn new(config: HeapConfig) -> Self {
        Heap {
            config,
            allocated_bytes: 0,
        }
    }

    /// Allocate memory on the heap.
    ///
    /// Returns an error if the allocation would exceed the memory limit.
    pub fn allocate(&mut self, bytes: usize) -> Result<(), JErrorType> {
        if let Some(max_bytes) = self.config.max_bytes {
            if self.allocated_bytes + bytes > max_bytes {
                return Err(JErrorType::RangeError("Out of memory".to_string()));
            }
        }
        self.allocated_bytes += bytes;
        Ok(())
    }

    /// Deallocate memory from the heap.
    pub fn deallocate(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
    }

    /// Get the current allocated bytes.
    pub fn get_allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Get the maximum allowed bytes, if any.
    pub fn get_max_bytes(&self) -> Option<usize> {
        self.config.max_bytes
    }

    /// Check if allocation of the given size would succeed.
    pub fn can_allocate(&self, bytes: usize) -> bool {
        if let Some(max_bytes) = self.config.max_bytes {
            self.allocated_bytes + bytes <= max_bytes
        } else {
            true
        }
    }

    /// Get the remaining available bytes, if limited.
    pub fn available_bytes(&self) -> Option<usize> {
        self.config
            .max_bytes
            .map(|max| max.saturating_sub(self.allocated_bytes))
    }
}

impl Default for Heap {
    fn default() -> Self {
        Self::new(HeapConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_unlimited() {
        let mut heap = Heap::new(HeapConfig::unlimited());
        assert!(heap.allocate(1000).is_ok());
        assert!(heap.allocate(1000000).is_ok());
        assert_eq!(heap.get_allocated(), 1001000);
    }

    #[test]
    fn test_heap_limited() {
        let mut heap = Heap::new(HeapConfig::with_limit(1000));
        assert!(heap.allocate(500).is_ok());
        assert!(heap.allocate(400).is_ok());
        assert_eq!(heap.get_allocated(), 900);

        // This should fail
        let result = heap.allocate(200);
        assert!(result.is_err());
        if let Err(JErrorType::RangeError(msg)) = result {
            assert_eq!(msg, "Out of memory");
        }
    }

    #[test]
    fn test_heap_deallocate() {
        let mut heap = Heap::new(HeapConfig::with_limit(1000));
        heap.allocate(500).unwrap();
        heap.deallocate(300);
        assert_eq!(heap.get_allocated(), 200);

        // Now we should be able to allocate more
        assert!(heap.allocate(700).is_ok());
    }

    #[test]
    fn test_heap_can_allocate() {
        let heap = Heap::new(HeapConfig::with_limit(1000));
        assert!(heap.can_allocate(500));
        assert!(heap.can_allocate(1000));
        assert!(!heap.can_allocate(1001));
    }

    #[test]
    fn test_heap_available_bytes() {
        let mut heap = Heap::new(HeapConfig::with_limit(1000));
        assert_eq!(heap.available_bytes(), Some(1000));
        heap.allocate(300).unwrap();
        assert_eq!(heap.available_bytes(), Some(700));

        let unlimited = Heap::new(HeapConfig::unlimited());
        assert_eq!(unlimited.available_bytes(), None);
    }
}
