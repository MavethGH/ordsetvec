pub use iter_set::Inclusion;
pub use std::cmp::Ordering;
use std::ops::Deref;

use thiserror::Error;

/// Must be implemented by all types in an `OrdSetVec`
/// Implemented for all types that are Ord
pub trait OrdSetItemTrait {
    type Key;
    fn compare(a: &Self, b: &Self) -> Ordering;
    fn compare_to(a: &Self, b: &Self::Key) -> Ordering;
}

impl<T: Ord> OrdSetItemTrait for T {
    type Key = T;

    fn compare(a: &Self, b: &Self) -> Ordering {
        a.cmp(b)
    }

    fn compare_to(a: &Self, b: &Self::Key) -> Ordering {
        a.cmp(b)
    }
}

/// Errors that can occur when creating or verifying `OrdSetIter` or `OrdSetVec`
#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("duplicate data")]
    DuplicateData,
    #[error("unsorted or duplicate data")]
    UnsortedOrDuplicate,
    #[error("item must compare greater than all items in the set")]
    ItemTooSmall,
    #[error("key not found")]
    NotFound,
}

pub type Result<T> = std::result::Result<T, VerificationError>;

/// A collection of *unique, sorted* values with very fast iteration
/// (as fast as `Vec`, cache-friendly), O(log(n)) lookup using binary search,
/// and poor insert/delete performance (O(n), just like a `Vec`)

pub struct OrdSetVec<T: OrdSetItemTrait> {
    inner: Vec<T>,
}

impl<T: OrdSetItemTrait> Default for OrdSetVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: OrdSetItemTrait> Deref for OrdSetVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: OrdSetItemTrait> OrdSetVec<T> {
    /// Creates a new, empty `OrdSetVec`
    ///
    /// See [`Vec:new()`]
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Constructs a new, empty `OrdSetVec` with at least the specified capacity.
    ///
    /// See [`Vec::with_capacity()`]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    /// Creates an `OrdSetVec` from an already sorted vector, verifying that it is indeed
    /// sorted and that there are no duplicate items.
    ///
    /// Returns an error if the Vec is not sorted or contains duplicates.
    pub fn from_sorted_vec(external: Vec<T>) -> Result<Self> {
        for i in 1..external.len() {
            match T::compare(&external[i - 1], &external[i]) {
                Ordering::Less => continue,
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => return Err(VerificationError::UnsortedOrDuplicate),
            }
        }

        Ok(Self::from_sorted_vec_unchecked(external))
    }

    /// Creates an `OrdSetVec` from an already sorted vector without any safeguards
    pub fn from_sorted_vec_unchecked(external: Vec<T>) -> Self {
        Self { inner: external }
    }

    /// Creates an `OrdSetVec` from an unsorted iterator
    ///
    /// Currently collects the iterator into a `Vec` and passes it to
    /// [`from_unsorted_vec()`](`Self::from_unsorted_vec()`)
    ///
    pub fn from_unsorted<I: Iterator<Item = T>>(iter: I) -> Result<Self> {
        Self::from_unsorted_vec(iter.collect())
    }

    /// Creates an `OrdSetVec` from an unsorted `Vec`.
    ///
    /// Use this
    ///
    /// Returns [`VerificationError::DuplicateData`] if the `Vec` contains duplicates
    pub fn from_unsorted_vec(mut external: Vec<T>) -> Result<Self> {
        external.sort_unstable_by(T::compare);

        match Self::find_dup(&external) {
            None => Ok(Self::from_sorted_vec_unchecked(external)),
            Some(_) => Err(VerificationError::DuplicateData),
        }
    }

    /// Appends an element to the back of the `OrdSetVec`.
    ///
    /// Returns an error if the item doesn't compare greater than the previous end item
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.inner.is_empty() {
            self.inner.push(item);
        } else {
            match T::compare(&item, self.inner.last().unwrap()) {
                Ordering::Less => return Err(VerificationError::ItemTooSmall),
                Ordering::Equal => return Err(VerificationError::DuplicateData),
                Ordering::Greater => self.inner.push(item),
            }
        }

        Ok(())
    }

    /// Most `Vec` methods that make sense to use on an `OrdSetVec` are already
    /// implemented on it.
    ///
    /// Use this if you need to pass an `OrdSetVec` to something that needs a `Vec`
    pub fn as_vec(&self) -> &Vec<T> {
        &self.inner
    }

    /// Same as [`Self::as_vec()`], but returns a mutable reference.
    ///
    /// Use with caution, since mutating the resulting `Vec` can easily break
    /// the `OrdSetVec` guarantees of being sorted and deduplicated. If you
    /// need to use this but can't prevent that, use [`Self::verify()`] before
    /// further using this `OrdSetVec`, to ensure that it is valid.
    pub fn as_vec_mut_unchecked(&mut self) -> &mut Vec<T> {
        &mut self.inner
    }

    /// Converts this `OrdSetVec` into a normal `Vec`, consuming it.
    ///
    /// Use this if you no longer need the guarantees of `OrdSetVec` and just want your
    /// data as a normal `Vec`.
    pub fn into_vec(self) -> Vec<T> {
        self.inner
    }

    /// Sorts this `OrdSetVec` and removes any duplicates. Call this after using any
    /// `..._unchecked` method if it possibly invalidated the contents of this struct
    pub fn verify(&mut self) {
        self.inner.sort_unstable_by(T::compare);
        self.inner
            .dedup_by(|a, b| T::compare(a, b) == Ordering::Equal);
    }

    /// Return the index of item in this `OrdSetVec` if it exists, by binary search.
    ///
    /// See [`slice::binary_search()`] for more details.
    pub fn binary_search_item(&self, item: &T) -> std::result::Result<usize, usize> {
        self.inner.binary_search_by(|e| T::compare(e, item))
    }

    /// Returns true if the item exists in the is `OrdSetVec`, and false if it doesn't.
    ///
    /// Uses [`Self::binary_search_item()`] internally.
    pub fn contains_item(&self, item: &T) -> bool {
        self.binary_search_item(item).is_ok()
    }

    /// Returns the index of key in this `OrdSetVec` if it exists
    pub fn binary_search_key(&self, key: &T::Key) -> std::result::Result<usize, usize> {
        self.inner.binary_search_by(|e| T::compare_to(e, key))
    }

    fn find_dup(slice: &[T]) -> Option<usize> {
        for i in 1..slice.len() {
            if T::compare(&slice[i - 1], &slice[i]) == Ordering::Equal {
                return Some(i);
            }
        }
        None
    }
}
